#include <iostream>

#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

template <typename T>
__global__ void testPrefixScan(uint32_t size) {
  __shared__ T ws[32];
  __shared__ T c[1024];
  __shared__ T co[1024];

  auto first = threadIdx.x;
  for (auto i = first; i < size; i += blockDim.x)
    c[i] = 1;
  __syncthreads();

  blockPrefixScan(c, co, size, ws);
  blockPrefixScan(c, size, ws);

  assert(1 == c[0]);
  assert(1 == co[0]);
  for (auto i = first + 1; i < size; i += blockDim.x) {
    if (c[i] != c[i - 1] + 1)
      printf("failed %d %d %d: %d %d\n", size, i, blockDim.x, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

template <typename T>
__global__ void testWarpPrefixScan(uint32_t size) {
  assert(size <= 32);
  __shared__ T c[1024];
  __shared__ T co[1024];
  auto i = threadIdx.x;
  c[i] = 1;
  __syncthreads();

  warpPrefixScan(c, co, i, 0xffffffff);
  warpPrefixScan(c, i, 0xffffffff);
  __syncthreads();

  assert(1 == c[0]);
  assert(1 == co[0]);
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      printf("failed %d %d %d: %d %d\n", size, i, blockDim.x, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}

__global__ void init(uint32_t *v, uint32_t val, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i] = val;
  if (i == 0)
    printf("init\n");
}

__global__ void verify(uint32_t const *v, uint32_t n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    assert(v[i] == i + 1);
  if (i == 0)
    printf("verify\n");
}

int main() {
  exitSansCUDADevices();

  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;
  testWarpPrefixScan<int><<<1, 32>>>(32);
  cudaDeviceSynchronize();
  // std::cout << "warp 16" << std::endl;
  testWarpPrefixScan<int><<<1, 32>>>(16);
  cudaDeviceSynchronize();
  // std::cout << "warp 5" << std::endl;
  testWarpPrefixScan<int><<<1, 32>>>(5);
  cudaDeviceSynchronize();

  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= 1024; bs += 32) {
    // std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // std::cout << j << std::endl;
      testPrefixScan<uint16_t><<<1, bs>>>(j);
      cudaDeviceSynchronize();
      testPrefixScan<float><<<1, bs>>>(j);
      cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblok" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    cudaCheck(cudaMalloc(&d_in, num_items * sizeof(uint32_t)));
    cudaCheck(cudaMalloc(&d_out1, num_items * sizeof(uint32_t)));
    cudaCheck(cudaMalloc(&d_out2, num_items * sizeof(uint32_t)));

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    init<<<nblocks, nthreads, 0>>>(d_in, 1, num_items);

    // the block counter
    int32_t *d_pc;
    cudaCheck(cudaMalloc(&d_pc, sizeof(int32_t)));
    cudaCheck(cudaMemset(d_pc, 0, 4));

    nthreads = 1024;
    nblocks = (num_items + nthreads - 1) / nthreads;
    multiBlockPrefixScan<<<nblocks, nthreads, 0>>>(d_in, d_out1, num_items, d_pc);
    verify<<<nblocks, nthreads, 0>>>(d_out1, num_items);
    cudaDeviceSynchronize();

    // test cub
    std::cout << "cub" << std::endl;
    // Determine temporary device storage requirements for inclusive prefix sum
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out2, num_items);

    std::cout << "temp storage " << temp_storage_bytes << std::endl;

    // Allocate temporary storage for inclusive prefix sum
    // fake larger ws already available
    temp_storage_bytes *= 8;
    cudaCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    std::cout << "temp storage " << temp_storage_bytes << std::endl;
    // Run inclusive prefix sum
    CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out2, num_items));
    std::cout << "temp storage " << temp_storage_bytes << std::endl;

    verify<<<nblocks, nthreads, 0>>>(d_out2, num_items);
    cudaDeviceSynchronize();
  }  // ksize
  return 0;
}
