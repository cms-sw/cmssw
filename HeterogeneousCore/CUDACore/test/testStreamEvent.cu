/**
 * The purpose of this test program is to ensure that the logic for
 * CUDA event use in cms::cuda::Product and cms::cuda::ScopedContext
 */

#include <iostream>
#include <memory>
#include <type_traits>
#include <chrono>
#include <thread>
#include <cassert>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace {
  constexpr int ARRAY_SIZE = 20000000;
  constexpr int NLOOPS = 10;
}  // namespace

__global__ void kernel_looping(float *point, unsigned int num) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  for (int iloop = 0; iloop < NLOOPS; ++iloop) {
    for (size_t offset = idx; offset < num; offset += gridDim.x * blockDim.x) {
      point[offset] += 1;
    }
  }
}

int main() {
  cms::cudatest::requireDevices();

  constexpr bool debug = false;

  float *dev_points1;
  float *host_points1;
  cudaStream_t stream1, stream2;
  cudaEvent_t event1, event2;

  cudaCheck(cudaMalloc(&dev_points1, ARRAY_SIZE * sizeof(float)));
  cudaCheck(cudaMallocHost(&host_points1, ARRAY_SIZE * sizeof(float)));
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
  cudaEventCreate(&event1);
  cudaEventCreate(&event2);

  for (size_t j = 0; j < ARRAY_SIZE; ++j) {
    host_points1[j] = static_cast<float>(j);
  }

  cudaCheck(cudaMemcpyAsync(dev_points1, host_points1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream1));
  kernel_looping<<<1, 16, 0, stream1>>>(dev_points1, ARRAY_SIZE);
  if (debug)
    std::cout << "Kernel launched on stream1" << std::endl;

  auto status = cudaStreamQuery(stream1);
  if (debug)
    std::cout << "Stream1 busy? " << (status == cudaErrorNotReady) << " idle? " << (status == cudaSuccess) << std::endl;
  cudaEventRecord(event1, stream1);
  status = cudaEventQuery(event1);
  if (debug)
    std::cout << "Event1 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  assert(status == cudaErrorNotReady);

  status = cudaStreamQuery(stream2);
  if (debug)
    std::cout << "Stream2 busy? " << (status == cudaErrorNotReady) << " idle? " << (status == cudaSuccess) << std::endl;
  assert(status == cudaSuccess);
  if (debug) {
    cudaEventRecord(event2, stream2);
    status = cudaEventQuery(event2);
    std::cout << "Event2 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    status = cudaEventQuery(event2);
    std::cout << "Event2 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  }

  cudaStreamWaitEvent(stream2, event1, 0);
  if (debug)
    std::cout << "\nStream2 waiting for event1" << std::endl;
  status = cudaStreamQuery(stream2);
  if (debug)
    std::cout << "Stream2 busy? " << (status == cudaErrorNotReady) << " idle? " << (status == cudaSuccess) << std::endl;
  assert(status == cudaErrorNotReady);
  cudaEventRecord(event2, stream2);
  status = cudaEventQuery(event2);
  if (debug)
    std::cout << "Event2 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  assert(status == cudaErrorNotReady);
  if (debug) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    status = cudaEventQuery(event2);
    std::cout << "Event2 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  }

  status = cudaStreamQuery(stream1);
  if (debug) {
    std::cout << "\nStream1 busy? " << (status == cudaErrorNotReady) << " idle? " << (status == cudaSuccess)
              << std::endl;
    std::cout << "Synchronizing stream1" << std::endl;
  }
  assert(status == cudaErrorNotReady);
  cudaStreamSynchronize(stream1);
  if (debug)
    std::cout << "Synchronized stream1" << std::endl;

  status = cudaEventQuery(event1);
  if (debug)
    std::cout << "Event1 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  assert(status == cudaSuccess);
  status = cudaEventQuery(event2);
  if (debug)
    std::cout << "Event2 recorded? " << (status == cudaErrorNotReady) << " occurred? " << (status == cudaSuccess)
              << std::endl;
  assert(status == cudaSuccess);

  cudaFree(dev_points1);
  cudaFreeHost(host_points1);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaEventDestroy(event1);
  cudaEventDestroy(event2);

  return 0;
}
