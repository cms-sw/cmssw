// MPI Test Worker process implementation (runs on rank != 0)
#include <utility>
#include <vector>

#include <mpi.h>

#include "MPITestBase.h"

class MPI_Worker : public MPI_TEST {
public:
  MPI_Worker() {
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    // assert(rank_ != 0 && size_ >= 2);
  }

  // Beginning of blockingSend (Worker)
  std::pair<float, float> blockingSend() override {
    // Root rank is always zero

    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_FLOAT, &messageSize);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    MPI_Recv(&v1[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&v2[0], messageSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    MPI_Send(&v1[0], v1.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of blockingSend

  // Beginning of nonBlockingSend (Worker)
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
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of nonBlockingSend

  // Beginning of blockingScatter (Worker)
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
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of blockingScatter

  // Beginning of nonBlockingScatter (Worker)
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
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of nonBlockingScatter

  // Beginning of blockingSendRecv (Worker)
  std::pair<float, float> blockingSendRecv() override {
    // Root rank is always zero

    MPI_Status status;
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_FLOAT, &messageSize);

    // Resize vectors
    v1.resize(messageSize);
    v2.resize(messageSize);

    // Receiving v1
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

    // Receiving v2
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

    // Adding the two vectors and the result is in v1
    for (size_t i = 0; i < v1.size(); i++) {
      v1[i] += v2[i];
    }

    // Sending back the result of the summation
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
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of blockingSendRecv

  // Beginning of oneSidedCommMaster (Worker side)
  std::pair<float, float> oneSidedCommMaster() override {
    // Root rank is always zero
    MPI_Win win1, win2;
    int vec_size;
    // Receiving the size of vectors
    MPI_Recv(&vec_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    v1.resize(vec_size);
    v2.resize(vec_size);

    // Window buffer
    /////////////////////////// Creating window //////////////////////////
    MPI_Win_create(&v1[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    //fence 1
    MPI_Win_fence(0, win1);

    MPI_Win_create(&v2[0], vec_size * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    // Fence 2
    MPI_Win_fence(0, win2);
    ////////////////////////// End of creating the window ///////////////
    //////////////////////// Getting the vector ////////////////
    // Fence 3
    MPI_Win_fence(0, win1);  // Put the double vector (v1 & v2) from root
    // Fence 4
    MPI_Win_fence(0, win2);  // Put the double vector (v1 & v2) from root

    //////////////// Calculating the summation of 2 vectors //////////////////

    for (int i = 0; i < vec_size; i++) {
      v1[i] += v2[i];
    }
    /////////////////////// End of calculation ///////////////////////
    // Fence 5
    MPI_Win_fence(0, win1);  // Get from root: root is getting the data from worker's window

    // Fence 6
    MPI_Win_fence(0, win1);  // Get from root: root is getting the data from worker's window

    // Free the window
    MPI_Win_free(&win1);
    MPI_Win_free(&win2);
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of oneSidedCommMaster

  // Beginning of oneSidedCommWorker (Worker side)
  std::pair<float, float> oneSidedCommWorker() override {
    // Root rank is always zero
    int vec_size;
    // Receiving the size of vectors
    MPI_Recv(&vec_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int MPI_size;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    int MPI_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    int rem = vec_size % (MPI_size - 1);

    int batchSize = vec_size / (MPI_size - 1) + (rem >= MPI_rank ? 1 : 0);

    // Window buffer
    std::vector<float> v1, v2;
    v1.resize(batchSize);
    v2.resize(batchSize);

    /////////////////////////// Creating window //////////////////////////
    MPI_Win win1;
    MPI_Win_create(&v1[0], batchSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
    // Fence 1
    MPI_Win_fence(0, win1);

    MPI_Win win2;
    MPI_Win_create(&v2[0], batchSize * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
    // Fence 2
    MPI_Win_fence(0, win2);
    ////////////////////////// End of creating the window ///////////////

    int dp = batchSize * (MPI_rank - 1) + (rem >= MPI_rank ? 0 : rem);

    ///////////////////////// Getting the vector ////////////////
    // Getting v1 from root
    MPI_Get(&v1[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win1);
    // Fence 3
    MPI_Win_fence(0, win1);  // Put the double vector (v1 & v2) from root

    // Getting v2 from root
    MPI_Get(&v2[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win2);
    // Fence 4
    MPI_Win_fence(0, win2);  // Put the double vector (v1 & v2) from root

    //////////////// Calculating the summation of 2 vectors //////////////////
    for (int i = 0; i < batchSize; i++) {
      v1[i] += v2[i];
    }
    // Fence 5
    MPI_Win_fence(0, win1);

    /////////////////////// End of calculation ///////////////////////
    MPI_Put(&v1[0], batchSize, MPI_FLOAT, 0, dp, batchSize, MPI_FLOAT, win1);
    // Fence 6
    MPI_Win_fence(0, win1);  // Get from root: root is getting the data from worker's window

    // Free the window
    MPI_Win_free(&win1);
    MPI_Win_free(&win2);
    return std::pair<float, float>(0.0f, 0.0f);
  }  // End of oneSidedCommWorker

private:
  int size_;
  int rank_;
  int messageSize;  // Size for the worker vectors
  std::vector<float> v1;
  std::vector<float> v2;
};
