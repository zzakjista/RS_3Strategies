# RS_3Strategies
Transformer 구조의 추천시스템의 성능 향상을 위한 3가지 기법 연구

# INTRODUCTION
- BERT4Rec은 자연어 처리에서 뛰어난 성능을 보인 트랜스포머의 인코더 구조를 성공적으로 적용했으나, 성능 향상을 위한 주변적인 기술적 연구를 포함하지 않았다는 한계를 가진다.
- 본 연구에서는 BERT4Rec의 성능 향상을 위한 세 가지 전략으로 해당 모델에 적합한 데이터 증강 기법, 가중치 초기화, 비유클리드 공간에서의 임베딩을 제안한다

# METHODOLOGY 
## SELF-ATTENTION WEIGHT INITIALIZATION
- Xavier, he 등 Normal distribution의 분산을 조절하는 기존의 초기화 기법은 선행 정보(Transactino)를 담지 못하는 무작위적 초기화(Random Initialization)이다 다른 초기화 기법인 Transfer learning이 좋은 대안이나 추천시스템 도메인에서는 유저의 행동 데이터를 기반으로 하기 때문에 대형 언어 및 이미지 모델처럼 일반화하기 어려운 한계가 있다. 트랜스포머 모델은 셀프어텐션 레이어와 Feed Foward Network 등의 레이어들이 하나의 블럭을 구성한 형태이다. 데이터셋의 크기와 비례해 Optimal Point를 찾기 위한 연산이 많아지기 때문에 해당 연산들을 줄이기 위한 방법을 제안했다.

- AUTOENCODER를 활용해 전체 유저의 Transaction의 Latent Space를 학습한다.
- Latent space는 (user_dim x reduced_item_dim) 혹은 (item_dim x reduced_user_dim)이다
- 전자는 유저를 기준으로 아이템의 latent factor를 추출한 것이다.
- TF-IDF에서 사용하는 행렬분해 기법인 SVD를 이용해 Latent space를 분해한다
- SVD는 좌특이벡터, 고유값, 우특이벡터로 분해되어, 고유벡터와 고유값을 추출할 수 있는 특징이 있다.
- 각 고유벡터들을 추출하여 Self-attention layer의 Query, Key, Value의 가중치에 연결한다
- W_q = user_latent_vector
- W_k = item_latent_vector
- W_v = dot(user_latent_vector, item_latent_vector)

## HYPERBOLIC EMBEDDING
- 대부분의 Neural Network의 임베딩에서는 유클리드 공간에서 정의되는 벡터를 사용한다. 쌍곡선 공간(Hyperbolic Space)는 음의 곡률을 가진 왜곡된 공간이라는 점에서 유클리드 공간에 비해 풍부한 표현을 할 수 있다는 장점을 가진다.

- 기존 모델은 임베딩 벡터 공간이 유클리드 공간의 좌표로 설정되어있기 때문에 해당 벡터를 쌍곡선 좌표로 보내는 변환을 진행했다

## DATA AUGMENTATION
- 데이터 증강은 데이터 부족 문제를 해결할 수 있는 실용적 기법이다. 추천시스템을 구현하는 환경 역시 유저의 데이터가 부족하다는 가정 아래, 해당 분야에서 적용할 수 있는 unique한 증강 기법을 제안한다
- Sequential Recommender System에서 데이터 증강 기법을 제안한 Song(2022)은 학습 데이터에 노이즈 추가, 유사한 아이템 교체, 중복 아이템 추가 등의 기법을 제안했다.
- 하지만 개별 유저의 시퀀스 길이를 반영하지 못했기 때문에 해당 연구에서는 유저의 시퀀스 길이에 따라 데이터 증강 비율을 다르게 설정할 수 있는 파라미터 p를 제안한다.


## HOW TO RUN?
- templtes.py 에서 파라미터 지정
  ``` python main.py --template train_bert ```
  
