#include "DataFormats/Math/interface/choleskyInversion.h"

using namespace math::cholesky;

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <iomanip>
#include <memory>
#include <algorithm>
#include <chrono>
#include <random>

#include <cassert>
#include <iostream>
#include <limits>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

template <typename M, int N>
__global__ void invert(M* mm, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  auto& m = mm[i];

  printf("before %d %f %f %f\n", N, m(0, 0), m(1, 0), m(1, 1));

  invertNN(m, m);

  printf("after %d %f %f %f\n", N, m(0, 0), m(1, 0), m(1, 1));
}

template <typename M, int N>
__global__ void invertE(M* mm, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  auto& m = mm[i];

  printf("before %d %f %f %f\n", N, m(0, 0), m(1, 0), m(1, 1));

  m = m.inverse();

  printf("after %d %f %f %f\n", N, m(0, 0), m(1, 0), m(1, 1));
}

// generate matrices
template <class M>
void genMatrix(M& m) {
  using T = typename std::remove_reference<decltype(m(0, 0))>::type;
  int n = M::ColsAtCompileTime;
  std::mt19937 eng;
  // std::mt19937 eng2;
  std::uniform_real_distribution<T> rgen(0., 1.);

  // generate first diagonal elemets
  for (int i = 0; i < n; ++i) {
    double maxVal = i * 10000 / (n - 1) + 1;  // max condition is 10^4
    m(i, i) = maxVal * rgen(eng);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      double v = 0.3 * std::sqrt(m(i, i) * m(j, j));  // this makes the matrix pos defined
      m(i, j) = v * rgen(eng);
      m(j, i) = m(i, j);
    }
  }
}

template <int DIM>
using MXN = Eigen::Matrix<double, DIM, DIM>;

int main() {
  cms::cudatest::requireDevices();

  constexpr int DIM = 6;

  using M = MXN<DIM>;

  M m;

  genMatrix(m);

  printf("on CPU before %d %f %f %f\n", DIM, m(0, 0), m(1, 0), m(1, 1));

  invertNN(m, m);

  printf("on CPU after %d %f %f %f\n", DIM, m(0, 0), m(1, 0), m(1, 1));

  double* d;
  cudaMalloc(&d, sizeof(M));
  cudaMemcpy(d, &m, sizeof(M), cudaMemcpyHostToDevice);
  invert<M, DIM><<<1, 1>>>((M*)d, 1);
  cudaCheck(cudaDeviceSynchronize());
  invertE<M, DIM><<<1, 1>>>((M*)d, 1);
  cudaCheck(cudaDeviceSynchronize());

  return 0;
}
