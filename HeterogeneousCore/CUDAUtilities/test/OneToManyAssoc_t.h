#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>
#include <memory>

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
#endif

#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"
using cms::cuda::AtomicPairCounter;

constexpr uint32_t MaxElem = 64000;
constexpr uint32_t MaxTk = 8000;
constexpr uint32_t MaxAssocs = 4 * MaxTk;

#ifdef RUNTIME_SIZE
using Assoc = cms::cuda::OneToManyAssoc<uint16_t, -1, -1>;
using SmallAssoc = cms::cuda::OneToManyAssoc<uint16_t, 128, -1>;
using Multiplicity = cms::cuda::OneToManyAssoc<uint16_t, 8, MaxTk>;
#else
using Assoc = cms::cuda::OneToManyAssoc<uint16_t, MaxElem, MaxAssocs>;
using SmallAssoc = cms::cuda::OneToManyAssoc<uint16_t, 128, MaxAssocs>;
using Multiplicity = cms::cuda::OneToManyAssoc<uint16_t, 8, MaxTk>;
#endif

using TK = std::array<uint16_t, 4>;

__global__ void countMultiLocal(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < n; i += gridDim.x * blockDim.x) {
    __shared__ Multiplicity::CountersOnly local;
    if (threadIdx.x == 0)
      local.zero();
    __syncthreads();
    local.count(2 + i % 4);
    __syncthreads();
    if (threadIdx.x == 0)
      assoc->add(local);
  }
}

__global__ void countMulti(TK const* __restrict__ tk, Multiplicity* __restrict__ assoc, int32_t n) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < n; i += gridDim.x * blockDim.x)
    assoc->count(2 + i % 4);
}

__global__ void verifyMulti(Multiplicity* __restrict__ m1, Multiplicity* __restrict__ m2) {
  auto first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < m1->totOnes(); i += gridDim.x * blockDim.x)
    assert(m1->off[i] == m2->off[i]);
}

__global__ void count(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < 4 * n; i += gridDim.x * blockDim.x) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->count(tk[k][j]);
  }
}

__global__ void fill(TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = first; i < 4 * n; i += gridDim.x * blockDim.x) {
    auto k = i / 4;
    auto j = i - 4 * k;
    assert(j < 4);
    if (k >= n)
      return;
    if (tk[k][j] < MaxElem)
      assoc->fill(tk[k][j], k);
  }
}

__global__ void verify(Assoc* __restrict__ assoc) { assert(int(assoc->size()) < assoc->capacity()); }

template <typename Assoc>
__global__ void fillBulk(AtomicPairCounter* apc, TK const* __restrict__ tk, Assoc* __restrict__ assoc, int32_t n) {
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int k = first; k < n; k += gridDim.x * blockDim.x) {
    auto m = tk[k][3] < MaxElem ? 4 : 3;
    assoc->bulkFill(*apc, &tk[k][0], m);
  }
}

template <typename Assoc>
__global__ void verifyBulk(Assoc const* __restrict__ assoc, AtomicPairCounter const* apc) {
  if (int(apc->get().m) >= assoc->nOnes())
    printf("Overflow %d %d\n", apc->get().m, assoc->nOnes());
  assert(int(assoc->size()) < assoc->capacity());
}

template <typename Assoc>
__global__ void verifyFill(Assoc const* __restrict__ la, int n) {
  printf("assoc size %d\n", la->size());
  int imax = 0;
  long long ave = 0;
  int z = 0;
  for (int i = 0; i < n; ++i) {
    auto x = la->size(i);
    if (x == 0) {
      z++;
      continue;
    }
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(n));
  printf("found with %d elements %f %d %d\n", n, double(ave) / n, imax, z);
}

template <typename Assoc>
__global__ void verifyFinal(Assoc const* __restrict__ la, int N) {
  printf("assoc size %d\n", la->size());

  int imax = 0;
  long long ave = 0;
  for (int i = 0; i < N; ++i) {
    auto x = la->size(i);
    if (!(x == 4 || x == 3))
      printf("%d %d\n", i, x);
    assert(x == 4 || x == 3);
    ave += x;
    imax = std::max(imax, int(x));
  }
  assert(0 == la->size(N));
  printf("found with ave occupancy %f %d\n", double(ave) / N, imax);
}

template <typename T>
auto make_unique(std::size_t size) {
#ifdef __CUDACC__
  return cms::cuda::make_device_unique<T>(size, 0);
#else
  return std::make_unique<T>(size);
#endif
}

int main() {
#ifdef __CUDACC__
  cms::cudatest::requireDevices();
  auto current_device = cms::cuda::currentDevice();
#else
  // make sure cuda emulation is working
  std::cout << "cuda x's " << threadIdx.x << ' ' << blockIdx.x << ' ' << blockDim.x << ' ' << gridDim.x << std::endl;
  std::cout << "cuda y's " << threadIdx.y << ' ' << blockIdx.y << ' ' << blockDim.y << ' ' << gridDim.y << std::endl;
  std::cout << "cuda z's " << threadIdx.z << ' ' << blockIdx.z << ' ' << blockDim.z << ' ' << gridDim.z << std::endl;
  assert(threadIdx.x == 0);
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  assert(blockIdx.x == 0);
  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(blockDim.x == 1);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
#endif

  std::cout << "OneToManyAssoc " << sizeof(Assoc) << ' ' << Assoc::ctNOnes() << ' ' << Assoc::ctCapacity() << std::endl;
  std::cout << "OneToManyAssoc (small) " << sizeof(SmallAssoc) << ' ' << SmallAssoc::ctNOnes() << ' '
            << SmallAssoc::ctCapacity() << std::endl;

  std::mt19937 eng;

  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  std::vector<std::array<uint16_t, 4>> tr(N);

  // fill with "index" to element
  long long ave = 0;
  int imax = 0;
  auto n = 0U;
  auto z = 0U;
  auto nz = 0U;
  for (auto i = 0U; i < 4U; ++i) {
    auto j = 0U;
    while (j < N && n < MaxElem) {
      if (z == 11) {
        ++n;
        z = 0;
        ++nz;
        continue;
      }  // a bit of not assoc
      auto x = rdm(eng);
      auto k = std::min(j + x + 1, N);
      if (i == 3 && z == 3) {  // some triplets time to time
        for (; j < k; ++j)
          tr[j][i] = MaxElem + 1;
      } else {
        ave += x + 1;
        imax = std::max(imax, x);
        for (; j < k; ++j)
          tr[j][i] = n;
        ++n;
      }
      ++z;
    }
    assert(n <= MaxElem);
    assert(j <= N);
  }
  std::cout << "filled with " << n << " elements " << double(ave) / n << ' ' << imax << ' ' << nz << std::endl;

  auto a_d = make_unique<Assoc[]>(1);
  auto sa_d = make_unique<SmallAssoc[]>(1);
#ifdef __CUDACC__
  auto v_d = cms::cuda::make_device_unique<std::array<uint16_t, 4>[]>(N, nullptr);
  assert(v_d.get());
  cudaCheck(cudaMemcpy(v_d.get(), tr.data(), N * sizeof(std::array<uint16_t, 4>), cudaMemcpyHostToDevice));
#else
  auto v_d = tr.data();
#endif

  Assoc::Counter* a_st = nullptr;
  int a_n = MaxElem;

  Assoc::index_type* a_st2 = nullptr;
  SmallAssoc::index_type* sa_st2 = nullptr;
  int a_n2 = MaxAssocs;

// storage
#ifdef RUNTIME_SIZE
  auto a_st_d = make_unique<Assoc::Counter[]>(a_n);
  auto a_st2_d = make_unique<Assoc::index_type[]>(a_n2);
  auto sa_st2_d = make_unique<SmallAssoc::index_type[]>(a_n2);
  a_st = a_st_d.get();
  a_st2 = a_st2_d.get();
  sa_st2 = sa_st2_d.get();
#endif
  Assoc::View aView = {a_d.get(), a_st, a_st2, a_n, a_n2};
  launchZero(aView, 0);
  SmallAssoc::View saView = {sa_d.get(), nullptr, sa_st2, -1, a_n2};
  launchZero(saView, 0);

#ifdef __CUDACC__
  auto nThreads = 256;
  auto nBlocks = (4 * N + nThreads - 1) / nThreads;

  count<<<nBlocks, nThreads>>>(v_d.get(), a_d.get(), N);

  launchFinalize(aView, 0);
  verify<<<1, 1>>>(a_d.get());
  fill<<<nBlocks, nThreads>>>(v_d.get(), a_d.get(), N);
  verifyFill<<<1, 1>>>(a_d.get(), n);

#else
  count(v_d, a_d.get(), N);
  launchFinalize(aView);
  verify(a_d.get());
  fill(v_d, a_d.get(), N);
  verifyFill(a_d.get(), n);

#endif

  // now the inverse map (actually this is the ....)
  AtomicPairCounter* dc_d;
  AtomicPairCounter dc(0);

#ifdef __CUDACC__
  cudaCheck(cudaMalloc(&dc_d, sizeof(AtomicPairCounter)));
  cudaCheck(cudaMemset(dc_d, 0, sizeof(AtomicPairCounter)));
  nBlocks = (N + nThreads - 1) / nThreads;
  fillBulk<<<nBlocks, nThreads>>>(dc_d, v_d.get(), a_d.get(), N);
  finalizeBulk<<<nBlocks, nThreads>>>(dc_d, a_d.get());
  verifyBulk<<<1, 1>>>(a_d.get(), dc_d);

  cudaCheck(cudaMemcpy(&dc, dc_d, sizeof(AtomicPairCounter), cudaMemcpyDeviceToHost));
  verifyFinal<<<1, 1>>>(a_d.get(), N);

  cudaCheck(cudaMemset(dc_d, 0, sizeof(AtomicPairCounter)));
  fillBulk<<<nBlocks, nThreads>>>(dc_d, v_d.get(), sa_d.get(), N);
  finalizeBulk<<<nBlocks, nThreads>>>(dc_d, sa_d.get());
  verifyBulk<<<1, 1>>>(sa_d.get(), dc_d);

#else
  dc_d = &dc;
  fillBulk(dc_d, v_d, a_d.get(), N);
  finalizeBulk(dc_d, a_d.get());
  verifyBulk(a_d.get(), dc_d);

  verifyFinal(a_d.get(), N);

  AtomicPairCounter sdc(0);
  fillBulk(&sdc, v_d, sa_d.get(), N);
  finalizeBulk(&sdc, sa_d.get());
  verifyBulk(sa_d.get(), &sdc);

#endif

  std::cout << "final counter value " << dc.get().n << ' ' << dc.get().m << std::endl;

  // here verify use of block local counters
  auto m1_d = make_unique<Multiplicity[]>(1);
  auto m2_d = make_unique<Multiplicity[]>(1);

  Multiplicity::index_type* m1_st = nullptr;
  Multiplicity::index_type* m2_st = nullptr;
  int m_n = 0;

#ifdef RUNTIME_SIZE
  m_n = MaxTk;
  auto m1_st_d = make_unique<Multiplicity::index_type[]>(m_n);
  auto m2_st_d = make_unique<Multiplicity::index_type[]>(m_n);
  m1_st = m1_st_d.get();
  m2_st = m1_st_d.get();
#endif
  Multiplicity::View view1 = {m1_d.get(), nullptr, m1_st, -1, m_n};
  Multiplicity::View view2 = {m2_d.get(), nullptr, m2_st, -1, m_n};
  launchZero(view1, 0);
  launchZero(view2, 0);

#ifdef __CUDACC__
  nBlocks = (4 * N + nThreads - 1) / nThreads;
  countMulti<<<nBlocks, nThreads>>>(v_d.get(), m1_d.get(), N);
  countMultiLocal<<<nBlocks, nThreads>>>(v_d.get(), m2_d.get(), N);
  verifyMulti<<<1, Multiplicity::ctNOnes()>>>(m1_d.get(), m2_d.get());

  launchFinalize(view1, 0);
  launchFinalize(view2, 0);
  verifyMulti<<<1, Multiplicity::ctNOnes()>>>(m1_d.get(), m2_d.get());

  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
#else
  countMulti(v_d, m1_d.get(), N);
  countMultiLocal(v_d, m2_d.get(), N);
  verifyMulti(m1_d.get(), m2_d.get());

  launchFinalize(view1, 0);
  launchFinalize(view2, 0);
  verifyMulti(m1_d.get(), m2_d.get());
#endif
  return 0;
}
