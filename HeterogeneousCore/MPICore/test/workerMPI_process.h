//initializing the libraries needed
//WORKER//
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

class MPI_Worker : public MPI_TEST {
public:
  MPI_Worker() {
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    //assert(rank_ != 0  && size_ >= 2);
  }

  std::pair<float, float> blockingSend() override {
    //ROOT RANK IS ALWAYS ZERO

    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_FLOAT, &messageSize);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    MPI_Recv(&v1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&v2[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    MPI_Send(&v1[0], v1.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  //beginning of nonBlockingSend
  std::pair<float, float> nonBlockingSend() override {
    MPI_Request requestSend;
    MPI_Request requestRecv[1];
    MPI_Status status;

    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_FLOAT, &messageSize);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    MPI_Irecv(&v1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestRecv[0]);
    MPI_Irecv(&v2[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestRecv[1]);

    MPI_Waitall(2, requestRecv, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    MPI_Issend(&v1[0], v1.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestSend);
  }
  //end of nonBlockingSend

  //beginning of blockingScatter
  std::pair<float, float> blockingScatter() override {
    MPI_Recv(&messageSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &v1[0], messageSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &v2[0], messageSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    MPI_Gatherv(&v1[0], v1.size(), MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  //end of blockingScatter

  //begging of nonBlockingScatter
  std::pair<float, float> nonBlockingScatter() override {
    MPI_Recv(&messageSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    MPI_Request requestScatter[2];
    MPI_Request requestGather;

    MPI_Iscatterv(NULL, NULL, NULL, MPI_FLOAT, &v1[0], messageSize, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestScatter[0]);

    MPI_Iscatterv(NULL, NULL, NULL, MPI_FLOAT, &v2[0], messageSize, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestScatter[1]);

    MPI_Waitall(2, requestScatter, MPI_STATUS_IGNORE);  // Wait for scatter operations to complete.

    for (size_t i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    MPI_Igatherv(&v1[0], v1.size(), MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD, &requestGather);
  }
  //end of nonBlockingScatter

  //beginning of blockingSendRecv WORKER
  std::pair<float, float> blockingSendRecv() override {
    //ROOT RANK IS ALWAYS ZERO

    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_FLOAT, &messageSize);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    //receiving v1
    MPI_Sendrecv(
        /*sending buf*/ NULL,
        /*send count*/ 0,
        /*send type*/ MPI_FLOAT,
        /*dst rank*/ 0, /*worker*/
        /*send tag*/ 0, /*root tag*/

        /*recv buf*/ &v1[0],
        /*recv count*/ messageSize,
        /*recv type*/ MPI_FLOAT,
        /*src rank*/ 0, /*root*/
        /*recv tag*/ 0, /*worker tag*/
        /*Comm.*/ MPI_COMM_WORLD,
        /*status*/ MPI_STATUS_IGNORE);

    //receiving v2
    MPI_Sendrecv(
        /*sending buf*/ NULL,
        /*send count*/ 0,
        /*send type*/ MPI_FLOAT,
        /*dst rank*/ 0, /*worker*/
        /*send tag*/ 1, /*root tag*/

        /*recv buf*/ &v2[0],
        /*recv count*/ messageSize,
        /*recv type*/ MPI_FLOAT,
        /*src rank*/ 0, /*root*/
        /*recv tag*/ 1, /*worker tag*/
        /*Comm.*/ MPI_COMM_WORLD,
        /*status*/ MPI_STATUS_IGNORE

    );

    //adding the two vectors and the result is in v1
    for (int i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    //sending back the result of the summation
    MPI_Sendrecv(
        /*sending buf*/ &v1[0],
        /*send count*/ v1.size(),
        /*send type*/ MPI_FLOAT,
        /*dst*/ 0,      /*root*/
        /*send tag*/ 2, /*worker tag*/

        /*recv buf*/ NULL,
        /*recv count*/ 0,
        /*recv type*/ MPI_FLOAT,
        /*src*/ 0,      /*worker rank_*/
        /*recv tag*/ 2, /*root tag*/
        /*Comm.*/ MPI_COMM_WORLD,
        /*status*/ MPI_STATUS_IGNORE

    );
  }
  //end of blockingSend

  //beggining of one sided communincation
  std::pair<float, float> oneSidedCommMaster() override {  //WORKER
    //ROOT RANK IS ALWAYS ZERO
    MPI_Win win1, win2;
    int vec_size;
    //receiving the size of vectors
    MPI_Recv(&vec_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    v1.resize(vec_size);
    v2.resize(vec_size);

    //widow buffer
    /////////////////////////// creating window //////////////////////////
    MPI_Win_create(&v1[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    //fence 1
    MPI_Win_fence(0, win1);

    MPI_Win_create(&v2[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    //fence 2
    MPI_Win_fence(0, win2);
    ////////////////////////// end of creating the window ///////////////
    //////////////////////// getting the vector ////////////////
    //fence 3
    MPI_Win_fence(0, win1);  //put the double vector(v1 & v2) from root
    //fence 4
    MPI_Win_fence(0, win2);  //put the double vector(v1 & v2) from root

    //////////////// calculating the summation of 2 vectors //////////////////

    for (int i = 0; i < vec_size; i++) {
      v1[i] += v2[i];
    }
    /////////////////////// end of calculation ///////////////////////
    //fence 5
    MPI_Win_fence(0, win1);  //get from root: root is getting the data from workers window

    //fence 6
    MPI_Win_fence(0, win1);  //get from root: root is getting the data from workers window

    //free the window
    MPI_Win_free(&win1);
    MPI_Win_free(&win2);
  }
  std::pair<float, float> oneSidedCommWorker() override {
    //ROOT RANK IS ALWAYS ZERO
    int vec_size;
    //receiving the size of vectors
    MPI_Recv(&vec_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int MPI_size;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    int MPI_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    int rem = vec_size % (MPI_size - 1);

    int batchSize = vec_size / (MPI_size - 1) + (rem >= MPI_rank ? 1 : 0);

    //widow buffer
    std::vector<float> v1, v2;
    v1.resize(batchSize);
    v2.resize(batchSize);

    /////////////////////////// creating window //////////////////////////
    MPI_Win win1;
    MPI_Win_create(&v1[0], batchSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    //fence 1
    MPI_Win_fence(0, win1);

    MPI_Win win2;
    MPI_Win_create(&v2[0], batchSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    //fence 2
    MPI_Win_fence(0, win2);
    ////////////////////////// end of creating the window ///////////////

    int dp = batchSize * (MPI_rank - 1) + (rem >= MPI_rank ? 0 : rem);

    ///////////////////////// getting the vector ////////////////
    //getting v1 from root
    MPI_Get(&v1[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win1);
    //fence 3
    MPI_Win_fence(0, win1);  //put the double vector(v1 & v2) from root

    //getting v2 from root
    MPI_Get(&v2[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win2);
    //fence 4
    MPI_Win_fence(0, win2);  //put the double vector(v1 & v2) from root

    //////////////// calculating the summation of 2 vectors //////////////////
    for (int i = 0; i < batchSize; i++) {
      v1[i] += v2[i];
    }
    //fence 5
    MPI_Win_fence(0, win1);

    /////////////////////// end of calculation ///////////////////////
    MPI_Put(&v1[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win1);
    //fence 6
    MPI_Win_fence(0, win1);  //get from root: root is getting the data from workers window

    //free the window
    MPI_Win_free(&win1);
    MPI_Win_free(&win2);
  }

private:
  int size_;
  int rank_;
  int messageSize;  //size for the worker vects
  std::vector<float> v1;
  std::vector<float> v2;
};
