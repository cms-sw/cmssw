//  author: Felice Pantaleo, CERN, 2018
#include "../interface/GPUSimpleVector.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <new>

__global__ void vector_pushback(GPU::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->push_back(index);
}

__global__ void vector_reset(GPU::SimpleVector<int> *foo) {

  foo->reset();
}

__global__ void vector_emplace_back(GPU::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->emplace_back(index);
}

int main() {
  auto maxN = 10000;
  GPU::SimpleVector<int> *obj_ptr = nullptr;
  GPU::SimpleVector<int> *d_obj_ptr = nullptr;
  GPU::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  bool success =
      cudaMallocHost(&obj_ptr, sizeof(GPU::SimpleVector<int>)) == cudaSuccess &&
      cudaMallocHost(&data_ptr, maxN * sizeof(int)) == cudaSuccess &&
      cudaMalloc(&d_data_ptr, maxN * sizeof(int)) == cudaSuccess;

  auto v = new (obj_ptr) GPU::SimpleVector<int>(maxN, data_ptr);

  cudaMallocHost(&tmp_obj_ptr, sizeof(GPU::SimpleVector<int>));
  new (tmp_obj_ptr) GPU::SimpleVector<int>(maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  success =
      success &&
      cudaMalloc(&d_obj_ptr, sizeof(GPU::SimpleVector<int>)) == cudaSuccess
      // ... and copy the object to the device.
      && cudaMemcpy(d_obj_ptr, tmp_obj_ptr, sizeof(GPU::SimpleVector<int>),
                    cudaMemcpyHostToDevice) == cudaSuccess;

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  assert(success);
  vector_pushback<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);

  cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>),
             cudaMemcpyDeviceToHost);

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN
                                 ? numBlocks * numThreadsPerBlock
                                 : maxN));
  vector_reset<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);

  cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>),
             cudaMemcpyDeviceToHost);

  assert(obj_ptr->size() == 0);

  vector_emplace_back<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);

  cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>),
             cudaMemcpyDeviceToHost);

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN
                                 ? numBlocks * numThreadsPerBlock
                                 : maxN));

  success = success and
            cudaMemcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int),
                       cudaMemcpyDeviceToHost) == cudaSuccess and
            cudaFreeHost(obj_ptr) == cudaSuccess and
            cudaFreeHost(data_ptr) == cudaSuccess and
            cudaFreeHost(tmp_obj_ptr) == cudaSuccess and
            cudaFree(d_data_ptr) == cudaSuccess and
            cudaFree(d_obj_ptr) == cudaSuccess;
  assert(success);
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
