#include <Eigen/Dense>

#include "HeterogeneousCore/CUDAUtilities/interface/eigenSoA.h"

template <int32_t S>
struct MySoA {
  // we can find a way to avoid this copy/paste???
  static constexpr int32_t stride() { return S; }

  eigenSoA::ScalarSoA<float, S> a;
  eigenSoA::ScalarSoA<float, S> b;
};

using V = MySoA<128>;

__global__ void testBasicSoA(float* p) {
  using namespace eigenSoA;

  assert(!isPowerOf2(0));
  assert(isPowerOf2(1));
  assert(isPowerOf2(1024));
  assert(!isPowerOf2(1026));

  using M3 = Eigen::Matrix<float, 3, 3>;

  __shared__ eigenSoA::MatrixSoA<M3, 64> m;

  int first = threadIdx.x + blockIdx.x * blockDim.x;
  if (0 == first)
    printf("before %f\n", p[0]);

  // a silly game...
  int n = 64;
  for (int i = first; i < n; i += blockDim.x * gridDim.x) {
    m[i].setZero();
    m[i](0, 0) = p[i];
    m[i](1, 1) = p[i + 64];
    m[i](2, 2) = p[i + 64 * 2];
  }
  __syncthreads();  // not needed

  for (int i = first; i < n; i += blockDim.x * gridDim.x)
    m[i] = m[i].inverse().eval();
  __syncthreads();

  for (int i = first; i < n; i += blockDim.x * gridDim.x) {
    p[i] = m[63 - i](0, 0);
    p[i + 64] = m[63 - i](1, 1);
    p[i + 64 * 2] = m[63 - i](2, 2);
  }

  if (0 == first)
    printf("after %f\n", p[0]);
}

#include <cassert>
#include <iostream>
#include <memory>
#include <random>

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#endif

int main() {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();
#endif

  float p[1024];

  std::uniform_real_distribution<float> rgen(0.01, 0.99);
  std::mt19937 eng;

  for (auto& r : p)
    r = rgen(eng);
  for (int i = 0, n = 64 * 3; i < n; ++i)
    assert(p[i] > 0 && p[i] < 1.);

  std::cout << p[0] << std::endl;
#ifdef __CUDACC__
  float* p_d;
  cudaCheck(cudaMalloc(&p_d, 1024 * 4));
  cudaCheck(cudaMemcpy(p_d, p, 1024 * 4, cudaMemcpyDefault));
  testBasicSoA<<<1, 1024>>>(p_d);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaMemcpy(p, p_d, 1024 * 4, cudaMemcpyDefault));
  cudaCheck(cudaDeviceSynchronize());
#else
  testBasicSoA(p);
#endif

  std::cout << p[0] << std::endl;

  for (int i = 0, n = 64 * 3; i < n; ++i)
    assert(p[i] > 1.);

  std::cout << "END" << std::endl;
  return 0;
}
