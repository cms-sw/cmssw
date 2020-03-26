// nvcc -O3 CholeskyDecomp_t.cu --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61 --compiler-options="-Ofast -march=native"
// add -DDOPROF to run  nvprof --metrics all

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "DataFormats/Math/interface/choleskyInversion.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"

constexpr int stride() { return 5 * 1024; }
template <int DIM>
using MXN = Eigen::Matrix<double, DIM, DIM>;
template <int DIM>
using MapMX = Eigen::Map<MXN<DIM>, 0, Eigen::Stride<DIM * stride(), stride()>>;

template <int N>
__global__ void invertSOA(double *__restrict__ p, unsigned int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  MapMX<N> m(p + i);
  math::cholesky::invert(m, m);
}

template <typename M, int N>
__global__ void invert(M *mm, unsigned int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  auto &m = mm[i];
  math::cholesky::invert(m, m);
}

template <typename M, int N>
__global__ void invertSeq(M *mm, unsigned int n) {
  if (threadIdx.x != 0)
    return;
  auto first = blockIdx.x * blockDim.x;
  auto last = std::min(first + blockDim.x, n);

  for (auto i = first; i < last; ++i) {
    auto &m = mm[i];
    math::cholesky::invert(m, m);
  }
}

// generate matrices
template <class M>
void genMatrix(M &m) {
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

template <int N>
void go(bool soa) {
  constexpr unsigned int DIM = N;
  using MX = MXN<DIM>;
  std::cout << "testing Matrix of dimension " << DIM << " size " << sizeof(MX) << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;
  auto delta1 = delta;
  auto delta2 = delta;

  constexpr unsigned int SIZE = 4 * 1024;

  MX mm[stride()];  // just storage in case of SOA
  double *__restrict__ p = (double *)(mm);

  if (soa) {
    for (unsigned int i = 0; i < SIZE; ++i) {
      MapMX<N> m(p + i);
      genMatrix(m);
    }
  } else {
    for (auto &m : mm)
      genMatrix(m);
  }

  std::cout << mm[SIZE / 2](1, 1) << std::endl;

  if (soa)
    for (unsigned int i = 0; i < SIZE; ++i) {
      MapMX<N> m(p + i);
      math::cholesky::invert(m, m);
      math::cholesky::invert(m, m);
    }
  else
    for (auto &m : mm) {
      math::cholesky::invert(m, m);
      math::cholesky::invert(m, m);
    }

  std::cout << mm[SIZE / 2](1, 1) << std::endl;

  auto m_d = cms::cuda::make_device_unique<double[]>(DIM * DIM * stride(), nullptr);
  cudaCheck(cudaMemcpy(m_d.get(), (double const *)(mm), stride() * sizeof(MX), cudaMemcpyHostToDevice));

  constexpr int NKK =
#ifdef DOPROF
      2;
#else
      1000;
#endif
  for (int kk = 0; kk < NKK; ++kk) {
    int threadsPerBlock = 128;
    int blocksPerGrid = SIZE / threadsPerBlock;

    delta -= (std::chrono::high_resolution_clock::now() - start);

    if (soa)
      cms::cuda::launch(invertSOA<DIM>, {blocksPerGrid, threadsPerBlock}, m_d.get(), SIZE);
    else
      cms::cuda::launch(invert<MX, DIM>, {blocksPerGrid, threadsPerBlock}, (MX *)(m_d.get()), SIZE);

    cudaCheck(cudaMemcpy(&mm, m_d.get(), stride() * sizeof(MX), cudaMemcpyDeviceToHost));

    delta += (std::chrono::high_resolution_clock::now() - start);

    if (0 == kk)
      std::cout << mm[SIZE / 2](1, 1) << std::endl;

    if (!soa) {
      delta1 -= (std::chrono::high_resolution_clock::now() - start);

#ifndef DOPROF
      cms::cuda::launch(invertSeq<MX, DIM>, {blocksPerGrid, threadsPerBlock}, (MX *)(m_d.get()), SIZE);
      cudaCheck(cudaMemcpy(&mm, m_d.get(), stride() * sizeof(MX), cudaMemcpyDeviceToHost));
#endif
      delta1 += (std::chrono::high_resolution_clock::now() - start);

      if (0 == kk)
        std::cout << mm[SIZE / 2](1, 1) << std::endl;
    }

    delta2 -= (std::chrono::high_resolution_clock::now() - start);
    if (soa)
#pragma GCC ivdep
      for (unsigned int i = 0; i < SIZE; ++i) {
        MapMX<N> m(p + i);
        math::cholesky::invert(m, m);
      }
    else
#pragma GCC ivdep
      for (auto &m : mm) {
        math::cholesky::invert(m, m);
      }

    delta2 += (std::chrono::high_resolution_clock::now() - start);
  }

  std::cout << mm[SIZE / 2](1, 1) << std::endl;

  double DNNK = NKK;
  std::cout << "cuda/cudaSeq/x86 computation took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / DNNK << ' '
            << std::chrono::duration_cast<std::chrono::milliseconds>(delta1).count() / DNNK << ' '
            << std::chrono::duration_cast<std::chrono::milliseconds>(delta2).count() / DNNK << ' ' << " ms"
            << std::endl;
}

int main() {
  cms::cudatest::requireDevices();

  go<2>(false);
  go<3>(false);
  go<4>(false);
  go<5>(false);
  go<6>(false);

  go<2>(true);
  go<3>(true);
  go<4>(true);
  go<5>(true);
  go<6>(true);

  // go<10>();
  return 0;
}
