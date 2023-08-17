//initializing the libraries needed
//ROOT//

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>
#include <unistd.h>
#include <cmath>  // for abs() from <cmath>
#include "processInterface.h"
class MPI_Master : public MPI_TEST {
public:
  MPI_Master(int vectorSize) {  //int comType, int vectorSize, int avg){

    generateRandomData(vectorSize);
  }

  //beginning of blockingSend
  std::pair<float, float> blockingSend() override {
    // Send input data from root process to worker processes.

    int batchSize = v1_.size() / (size_ - 1);     //the size for each process execluding root
    int extraBatches = v1_.size() % (size_ - 1);  //the size for the batches thatll get the extra (%) execluding root

    float startTime = MPI_Wtime();

    int curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      //data       //count   //type //dst rank//tag //comm.
      MPI_Send(&v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      curIdx += curSize;
    }

    float endTime = MPI_Wtime();

    float sendDuration = (endTime - startTime) * 1000;

    result_.resize(v1_.size());

    startTime = MPI_Wtime();

    curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      //dat            //count  //type //src rank//tag //comm.    //status
      MPI_Recv(&result_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      curIdx += curSize;
    }

    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;

    checkResult(result_);
    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of blockingSend

  //beginning of nonBlockingSend
  std::pair<float, float> nonBlockingSend() override {
    MPI_Request requestSend[2 * (size_ - 1)];  //two for each process (one for sending v1 and the other for sending v2)
    MPI_Request requestRecv[size_ - 1];

    int batchSize = v1_.size() / (size_ - 1);     //the size for each process execluding root
    int extraBatches = v1_.size() % (size_ - 1);  //the size for the batches thatll get the extra (%) execluding root

    float startTime = MPI_Wtime();

    int curIdx = 0;

    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      MPI_Issend(&v1_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestSend[(i - 1) * 2]);
      MPI_Issend(&v2_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestSend[(i - 1) * 2 + 1]);
      curIdx += curSize;
    }
    MPI_Waitall(2 * (size_ - 1), requestSend, MPI_STATUS_IGNORE);

    float endTime = MPI_Wtime();
    float sendDuration = (endTime - startTime) * 1000;

    result_.resize(v1_.size());

    startTime = MPI_Wtime();

    curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      MPI_Irecv(&result_[curIdx], curSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRecv[i - 1]);
      curIdx += curSize;
    }

    //wait for all receive processes
    for (int i = 1; i < size_; i++) {
      MPI_Wait(&requestRecv[i - 1], MPI_STATUS_IGNORE);
    }

    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;

    checkResult(result_);
    return std::pair<float, float>(sendDuration, recvDuration);
  }
  //end of nonBlockingSend

  // beginning of blockingScatter
  std::pair<float, float> blockingScatter() override {
    std::vector<int> numDataPerProcess_(size_, 0);
    std::vector<int> displacementIndices_(size_, 0);

    int batchSize = v1_.size() / (size_ - 1);     //the size for each process execluding root
    int extraBatches = v1_.size() % (size_ - 1);  //the size for the batches thatll get the extra (%) execluding root

    int curIdx = 0;
    for (int i = 1; i < size_; i++) {
      numDataPerProcess_[i] = batchSize + ((extraBatches >= i) ? 1 : 0);
      displacementIndices_[i] = curIdx;
      curIdx += batchSize + ((extraBatches >= i) ? 1 : 0);
    }

    for (int i = 1; i < size_; i++) {
      int sizeToSend = batchSize + ((extraBatches >= i) ? 1 : 0);
      MPI_Send(&sizeToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    float startTime = MPI_Wtime();
    // Start scattering.
    MPI_Scatterv(
        &v1_[0], &numDataPerProcess_[0], &displacementIndices_[0], MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(
        &v2_[0], &numDataPerProcess_[0], &displacementIndices_[0], MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float endTime = MPI_Wtime();
    result_.resize(v1_.size());

    float sendDuration = (endTime - startTime) * 1000;

    startTime = MPI_Wtime();

    MPI_Gatherv(
        NULL, 0, MPI_FLOAT, &result_[0], &numDataPerProcess_[0], &displacementIndices_[0], MPI_FLOAT, 0, MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;

    checkResult(result_);

    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of blockingScatter

  //beginnig of nonBlockingScatter
  std::pair<float, float> nonBlockingScatter() override {
    std::vector<int> numDataPerProcess_(size_, 0);
    std::vector<int> displacementIndices_(size_, 0);

    int batchSize = v1_.size() / (size_ - 1);     //the size for each process execluding root
    int extraBatches = v1_.size() % (size_ - 1);  //the size for the batches thatll get the extra (%) execluding root

    MPI_Request requestScatter[2];
    MPI_Request requestGather;

    int curIdx = 0;
    for (int i = 1; i < size_; i++) {
      numDataPerProcess_[i] = batchSize + ((extraBatches >= i) ? 1 : 0);
      displacementIndices_[i] = curIdx;
      curIdx += batchSize + ((extraBatches >= i) ? 1 : 0);
    }

    for (int i = 1; i < size_; i++) {
      int sizeToSend = batchSize + ((extraBatches >= i) ? 1 : 0);
      MPI_Send(&sizeToSend, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    float startTime = MPI_Wtime();  // Get the start time before scattering.

    MPI_Iscatterv(&v1_[0],
                  &numDataPerProcess_[0],
                  &displacementIndices_[0],
                  MPI_FLOAT,
                  NULL,
                  0,
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD,
                  &requestScatter[0]);

    MPI_Iscatterv(&v2_[0],
                  &numDataPerProcess_[0],
                  &displacementIndices_[0],
                  MPI_FLOAT,
                  NULL,
                  0,
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD,
                  &requestScatter[1]);

    float endTime = MPI_Wtime();
    result_.resize(v1_.size());

    float sendDuration = (endTime - startTime) * 1000;

    startTime = MPI_Wtime();
    MPI_Igatherv(NULL,
                 0,
                 MPI_FLOAT,
                 &result_[0],
                 &numDataPerProcess_[0],
                 &displacementIndices_[0],
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD,
                 &requestGather);

    MPI_Wait(&requestGather, MPI_STATUS_IGNORE);  // Wait for gather operation to complete.

    endTime = MPI_Wtime();

    float recvDuration = (endTime - startTime) * 1000;

    checkResult(result_);

    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of nonBlockingScatter

  //beginning of blockingSendRecv ROOT
  std::pair<float, float> blockingSendRecv() override {
    // Send input data from root process to worker processes.
    int batchSize = v1_.size() / (size_ - 1);  //the size for each process execluding root

    //the size for the batches thatll get the extra (%) execluding root
    int extraBatches = v1_.size() % (size_ - 1);

    float startTime = MPI_Wtime();

    int curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch

      MPI_Sendrecv(  //sending v1
          /*sending buf*/ &v1_[curIdx],
          /*send count*/ curSize,
          /*send type*/ MPI_FLOAT,
          /*dst rank*/ i, /*worker*/
          /*send tag*/ 0, /*root tag*/

          /*recv buf*/ NULL, /*not receiving*/
          /*recv count*/ 0,
          /*recv type*/ MPI_FLOAT,
          /*src rank*/ i, /*root*/
          /*recv tag*/ 0, /*worker tag*/
          /*Comm.*/ MPI_COMM_WORLD,
          /*status*/ MPI_STATUS_IGNORE

      );

      MPI_Sendrecv(  //sending v2
          /*sending buf*/ &v2_[curIdx],
          /*send count*/ curSize,
          /*send type*/ MPI_FLOAT,
          /*dst*/ i, /*worker*/
          /*send tag*/ 1,

          /*recv buf*/ NULL,
          /*recv count*/ 0,
          /*recv type*/ MPI_FLOAT,
          /*src*/ i, /*root*/
          /*recv tag*/ 1,
          /*Comm.*/ MPI_COMM_WORLD,
          /*status*/ MPI_STATUS_IGNORE

      );

      curIdx += curSize;
    }

    float endTime = MPI_Wtime();

    float sendDuration = (endTime - startTime) * 1000;

    result_.resize(v1_.size());

    startTime = MPI_Wtime();

    curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch

      //receiving the result_ of adding v1 to v2
      MPI_Sendrecv(
          /*sending buf*/ NULL,
          /*send count*/ 0,
          /*send type*/ MPI_FLOAT,
          /*dst*/ i,      /*root*/
          /*send tag*/ 2, /*worker tag*/

          /*recv buf*/ &result_[curIdx],
          /*recv count*/ curSize,
          /*recv type*/ MPI_FLOAT,
          /*src*/ i,      /*worker*/
          /*recv tag*/ 2, /*root tag*/
          /*Comm.*/ MPI_COMM_WORLD,
          /*status*/ MPI_STATUS_IGNORE

      );

      curIdx += curSize;
    }

    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;

    checkResult(result_);
    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of blockingSendRecv

  //beginning of oneSidedComm
  std::pair<float, float> oneSidedCommMaster() override {  //ROOT
    int vec_size = v1_.size();
    int batchSize = v1_.size() / (size_ - 1);  //the size for each process execluding root
    //the size for the batches thatll get the extra (%) execluding root
    int extraBatches = v1_.size() % (size_ - 1);
    //sending the size to multi workers
    int curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      // sending vec size on chunks
      MPI_Send(&curSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      curIdx += curSize;
    }

    //resize the result_ vec
    result_.resize(v1_.size());

    //////////////////// creating a window //////////////////////
    MPI_Win win1;
    MPI_Win_create(&v1_[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    //fence 1
    MPI_Win_fence(0, win1);

    MPI_Win win2;
    MPI_Win_create(&v2_[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    //fence 2
    MPI_Win_fence(0, win2);
    //////////////////// end creating window ///////////////////
    //starting the time of sending
    float startTime = MPI_Wtime();

    ////////////////////// sending the vectors to workers /////////////////////////
    curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      MPI_Put(&v1_[curIdx], curSize, MPI_FLOAT, i, 0, curSize, MPI_FLOAT, win1);
      MPI_Put(&v2_[curIdx], curSize, MPI_FLOAT, i, 0, curSize, MPI_FLOAT, win2);
      curIdx += curSize;
    }
    //fence 3
    MPI_Win_fence(0, win1);
    //fence 4
    MPI_Win_fence(0, win2);

    ///////////////////////// end of sending the data /////////////////////////////

    //ending the time
    float endTime = MPI_Wtime();
    float sendDuration = (endTime - startTime) * 1000;

    ///////////////////////// getting the data from workers ////////////////////////

    //starting the time for receiving
    startTime = MPI_Wtime();

    //fence 5
    MPI_Win_fence(0, win1);

    //getting the result_ from worker
    curIdx = 0;
    for (int i = 1; i < size_; i++) {
      int curSize = batchSize + ((extraBatches >= i) ? 1 : 0);  //size of cur Batch
      //getting result_s from multi workers
      MPI_Get(&result_[curIdx], curSize, MPI_FLOAT, i, 0, curSize, MPI_FLOAT, win1);
      curIdx += curSize;
    }

    //fence 6
    MPI_Win_fence(0, win1);
    //ending the time of receiving
    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;

    //////////////////////////// ending getting the data ////////////////////////////

    checkResult(result_);  //check result_
    MPI_Win_free(&win1);   //freeing the window
    MPI_Win_free(&win2);   //freeing the window
    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of oneSidedComm

  std::pair<float, float> oneSidedCommWorker() override {
    // Send input data from root process to worker processes.
    // sending vec size
    int vec_size = v1_.size();
    //sending the size to multi workers
    for (int i = 1; i < size_; i++)
      MPI_Send(&vec_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

    //resize the result_ vec
    result_.resize(v1_.size());
    for (int i = 0; i < v1_.size(); i++) {
      result_[i] = v1_[i];
    }
    //////////////////// creating a window //////////////////////
    MPI_Win win1;
    MPI_Win_create(&v1_[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    //fence 1
    MPI_Win_fence(0, win1);
    MPI_Win win2;
    MPI_Win_create(&v2_[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    //fence 2
    MPI_Win_fence(0, win2);
    //////////////////// end creating window ///////////////////

    //starting the time of sending
    float startTime = MPI_Wtime();

    ////////////////////// worker gets data from root  /////////////////////////
    //fence 3
    MPI_Win_fence(0, win1);
    //fence 4
    MPI_Win_fence(0, win2);
    ///////////////////////// end of sending the data /////////////////////////////

    //ending the time
    float endTime = MPI_Wtime();
    float sendDuration = (endTime - startTime) * 1000;

    ///////////////////////// getting the data from workers ////////////////////////

    //fence 5
    MPI_Win_fence(0, win1);  //unnecessary, I assume time measurement is more accurate this way.
    //starting the time for receiving
    startTime = MPI_Wtime();

    // worker to put data after processing in v1_
    //fence 6
    MPI_Win_fence(0, win1);

    //ending the time of receiving
    endTime = MPI_Wtime();
    float recvDuration = (endTime - startTime) * 1000;
    //////////////////////////// ending getting the data ////////////////////////////
    for (int i = 0; i < v1_.size(); i++) {
      float x = result_[i];
      result_[i] = v1_[i];
      v1_[i] = x;
    }
    checkResult(result_);  //check result_
    MPI_Win_free(&win1);   //freeing the window
    MPI_Win_free(&win2);   //freeing the window
    return std::pair<float, float>(sendDuration, recvDuration);

  }  //end of oneSidedComm
};
