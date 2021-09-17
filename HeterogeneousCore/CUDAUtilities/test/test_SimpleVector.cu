//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include <cuda.h>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

__global__ void vector_pushback(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->push_back(index);
}

__global__ void vector_reset(cms::cuda::SimpleVector<int> *foo) { foo->reset(); }

__global__ void vector_emplace_back(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->emplace_back(index);
}

int main() {
  cms::cudatest::requireDevices();

  auto maxN = 10000;
  cms::cuda::SimpleVector<int> *obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *d_obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  cudaCheck(cudaMallocHost(&obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cudaCheck(cudaMallocHost(&data_ptr, maxN * sizeof(int)));
  cudaCheck(cudaMalloc(&d_data_ptr, maxN * sizeof(int)));

  auto v = cms::cuda::make_SimpleVector(obj_ptr, maxN, data_ptr);

  cudaCheck(cudaMallocHost(&tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cms::cuda::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  cudaCheck(cudaMalloc(&d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  // ... and copy the object to the device.
  cudaCheck(cudaMemcpy(d_obj_ptr, tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), cudaMemcpyDefault));

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  vector_pushback<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), cudaMemcpyDefault));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  vector_reset<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), cudaMemcpyDefault));

  assert(obj_ptr->size() == 0);

  vector_emplace_back<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  cudaCheck(cudaMemcpy(obj_ptr, d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>), cudaMemcpyDefault));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  cudaCheck(cudaMemcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int), cudaMemcpyDefault));
  cudaCheck(cudaFreeHost(obj_ptr));
  cudaCheck(cudaFreeHost(data_ptr));
  cudaCheck(cudaFreeHost(tmp_obj_ptr));
  cudaCheck(cudaFree(d_data_ptr));
  cudaCheck(cudaFree(d_obj_ptr));
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
