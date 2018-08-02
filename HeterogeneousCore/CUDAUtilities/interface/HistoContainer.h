#ifndef HeterogeneousCore_CUDAUtilities_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_HistoContainer_h

#include <cassert>
#include <cstdint>
#include <algorithm>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif // __CUDA_ARCH__

#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#ifdef __CUDACC__
namespace cudautils {

  template<typename Histo>
  __global__
  void zeroMany(Histo * h, uint32_t nh) {
    auto i  = blockIdx.x * blockDim.x + threadIdx.x;
    auto ih = i / Histo::nbins();
    auto k  = i - ih * Histo::nbins();
    if (ih < nh) {
      h[ih].nspills = 0;
      if (k < Histo::nbins())
        h[ih].n[k] = 0;
    }
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h,  uint32_t nh, T const * v, uint32_t * offsets) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= offsets[nh]) return;
     auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
     assert((*off) > 0);
     int32_t ih = off - offsets - 1;
     assert(ih >= 0);
     assert(ih < nh); 
     h[ih].fill(v, i);
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h, T const * v, uint32_t size) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < size) h->fill(v, i);
  }

  template<typename Histo>
  void zero(Histo * h, uint32_t nh, int nthreads, cudaStream_t stream) {
    auto nblocks = (nh * Histo::nbins() + nthreads - 1) / nthreads;
    zeroMany<<<nblocks, nthreads, 0, stream>>>(h, nh);
    cudaCheck(cudaGetLastError());
  }

  template<typename Histo, typename T>
  void fillOneFromVector(Histo * h, T const * v, uint32_t size, int nthreads, cudaStream_t stream) {
    zero(h, 1, nthreads, stream);
    auto nblocks = (size + nthreads - 1) / nthreads;
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, v, size);
    cudaCheck(cudaGetLastError());
  }

  template<typename Histo, typename T>
  void fillManyFromVector(Histo * h, uint32_t nh, T const * v, uint32_t * offsets, uint32_t totSize, int nthreads, cudaStream_t stream) {
    zero(h, nh, nthreads, stream);
    auto nblocks = (totSize + nthreads - 1) / nthreads;
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
    cudaCheck(cudaGetLastError());
  }

} // namespace cudautils
#endif

template<typename T, uint32_t N, uint32_t M>
class HistoContainer {
public:
#ifdef __CUDACC__
  using Counter = uint32_t;
#else
  using Counter = std::atomic<uint32_t>;
#endif

  static constexpr uint32_t sizeT()     { return sizeof(T) * 8; }
  static constexpr uint32_t nbins()     { return 1 << N; }
  static constexpr uint32_t shift()     { return sizeT() - N; }
  static constexpr uint32_t mask()      { return nbins() - 1; }
  static constexpr uint32_t binSize()   { return 1 << M; }
  static constexpr uint32_t spillSize() { return 4 * binSize(); }

  static constexpr uint32_t bin(T t) {
    return (t >> shift()) & mask();
  }

  void zero() {
    nspills = 0;
    for (auto & i : n)
      i = 0;
  }

  static constexpr
  uint32_t atomicIncrement(Counter & x) {
    #ifdef __CUDA_ARCH__
    return atomicAdd(&x, 1);
    #else
    return x++;
    #endif
  }

  __host__ __device__
  void fill(T const * t, uint32_t j) {
    auto b = bin(t[j]);
    auto w = atomicIncrement(n[b]);
    if (w < binSize()) {
      bins[b * binSize() + w] = j;
    } else {
      auto w = atomicIncrement(nspills);
      if (w < spillSize())
        spillBin[w] = j;
    }     
  }

  constexpr bool fullSpill() const {
    return nspills >= spillSize();
  }

  constexpr bool full(uint32_t b) const {
    return n[b] >= binSize();
  }

  constexpr auto const * begin(uint32_t b) const {
     return bins + b * binSize();
  }

  constexpr auto const * end(uint32_t b) const {
     return begin(b) + std::min(binSize(), uint32_t(n[b]));
  }

  constexpr auto size(uint32_t b) const {
     return n[b];
  }

  uint32_t bins[nbins()*binSize()];
  Counter  n[nbins()];
  uint32_t spillBin[spillSize()];
  Counter  nspills;
};

#endif // HeterogeneousCore_CUDAUtilities_HistoContainer_h
