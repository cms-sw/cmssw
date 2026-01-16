// from https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>
#include <exception>
#include <memory>
#include <math.h>
#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace torchtest {

  __global__ void vector_add_kernel(int* a, int* b, int* c, int N) {
    int t_id = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (t_id < N) {
      c[t_id] = a[t_id] + b[t_id];
    }
  }

  void vector_add(int* a, int* b, int* c, int N, int cuda_grid_size, int cuda_block_size, cudaStream_t stream) {
    vector_add_kernel<<<cuda_grid_size, cuda_block_size, 0, stream>>>(a, b, c, N);
    cudaGetLastError();
  }

}  // namespace torchtest

int main(int argc, const char* argv[]) {
  // temporary workaround to disable test on non-CUDA devices
  if (not cms::cudatest::testDevices())
    return 0;

  cudaStream_t stream;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess)
    CPPUNIT_FAIL("cudaStreamCreate failed");

  // Setup array, here 2^16 = 65536 items
  const int N = 1 << 16;
  size_t bytes = N * sizeof(int);

  // Declare pinned memory pointers
  int *a_cpu, *b_cpu, *c_cpu;

  // Allocate pinned memory for the pointers
  cudaMallocHost(&a_cpu, bytes);
  cudaMallocHost(&b_cpu, bytes);
  cudaMallocHost(&c_cpu, bytes);

  // Init vectors
  for (int i = 0; i < N; ++i) {
    a_cpu[i] = rand() % 100;
    b_cpu[i] = rand() % 100;
  }

  // Declare GPU memory pointers
  int *a_gpu, *b_gpu, *c_gpu;

  // Allocate memory on the device
  cudaMallocAsync(&a_gpu, bytes, stream);
  cudaMallocAsync(&b_gpu, bytes, stream);
  cudaMallocAsync(&c_gpu, bytes, stream);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpyAsync(a_gpu, a_cpu, bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(b_gpu, b_cpu, bytes, cudaMemcpyHostToDevice, stream);

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  torchtest::vector_add(a_gpu, b_gpu, c_gpu, N, NUM_BLOCKS, NUM_THREADS, stream);

  try {
    // Convert pinned memory on GPU to Torch tensor on GPU
    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, 0).pinned_memory(true);
    torch::Tensor a_gpu_tensor = torch::from_blob(a_gpu, {N}, options);
    torch::Tensor b_gpu_tensor = torch::from_blob(b_gpu, {N}, options);
    torch::Tensor c_gpu_tensor = torch::from_blob(c_gpu, {N}, options);
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;

    cudaFreeHost(a_cpu);
    cudaFreeHost(b_cpu);
    cudaFreeHost(c_cpu);

    cudaFreeAsync(a_gpu, stream);
    cudaFreeAsync(b_gpu, stream);
    cudaFreeAsync(c_gpu, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 1;
  }

  cudaMemcpyAsync(c_cpu, c_gpu, bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < N; ++i) {
    assert(c_cpu[i] == a_cpu[i] + b_cpu[i]);
  }

  cudaFreeHost(a_cpu);
  cudaFreeHost(b_cpu);
  cudaFreeHost(c_cpu);

  cudaFreeAsync(a_gpu, stream);
  cudaFreeAsync(b_gpu, stream);
  cudaFreeAsync(c_gpu, stream);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}
