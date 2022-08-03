#include <iostream>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/maxCoopBlocks.h"

using namespace cms::cuda;

template <typename T>
struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

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
      printf(format_traits<T>::failed_msg, size, i, blockDim.x, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] == co[i]);
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
      printf(format_traits<T>::failed_msg, size, i, blockDim.x, c[i], c[i - 1]);
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

template <typename T>
__global__ void doCoop(T const *ici, T *ico, int32_t size, T *ipsum) {
  coopBlockPrefixScan(ici, ico, size, ipsum);
}

int main() {
  cms::cudatest::requireDevices();

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
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    std::cout << "multiblok " << num_items << std::endl;
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
    cudaCheck(cudaMemset(d_pc, 0, sizeof(int32_t)));

    nthreads = 1024;
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nblocks << std::endl;
    multiBlockPrefixScan<<<nblocks, nthreads, 4 * nblocks>>>(d_in, d_out1, num_items, d_pc);
    cudaCheck(cudaGetLastError());
    verify<<<nblocks, nthreads, 0>>>(d_out1, num_items);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    uint32_t *d_psum;
    cudaCheck(cudaMalloc(&d_psum, nblocks * sizeof(uint32_t)));
    std::cout << "launch coopBlockPrefixScan " << num_items << ' ' << nblocks << std::endl;
    int maxBlocks = maxCoopBlocks(doCoop<uint32_t>, nthreads, 0, 0, 0);
    std::cout << "max number of blocks is " << maxBlocks << std::endl;
    auto ncoopblocks = std::min(nblocks, maxBlocks);
    void *kernelArgs[] = {&d_in, &d_out2, &num_items, &d_psum};
    dim3 dimBlock(nthreads, 1, 1);
    dim3 dimGrid(ncoopblocks, 1, 1);
    // launch
    cudaLaunchCooperativeKernel((void *)doCoop<uint32_t>, dimGrid, dimBlock, kernelArgs);
    cudaCheck(cudaGetLastError());
    verify<<<nblocks, nthreads, 0>>>(d_out2, num_items);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());

    std::cout << "Free" << std::endl;
    cudaCheck(cudaFree(d_psum));
    cudaCheck(cudaFree(d_out2));
    cudaCheck(cudaFree(d_out1));
    cudaCheck(cudaFree(d_in));

  }  // ksize
  return 0;
}
