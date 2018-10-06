#ifndef HeterogeneousCore_CUDAUtilities_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_HistoContainer_h

#include <algorithm>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif // __CUDA_ARCH__
#include <cstddef> 
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#endif

#ifdef __CUDACC__
namespace cudautils {

  template<typename Histo, typename T>
  __global__
  void countFromVector(Histo * __restrict__ h,  uint32_t nh, T const * __restrict__ v, uint32_t const * __restrict__ offsets) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= offsets[nh]) return;
     auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
     assert((*off) > 0);
     int32_t ih = off - offsets - 1;
     assert(ih >= 0);
     assert(ih < nh);
     (*h).count(v[i], ih);
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * __restrict__ h,  uint32_t nh, T const * __restrict__ v, uint32_t const * __restrict__ offsets,
                      uint32_t * __restrict__ ws ) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= offsets[nh]) return;
     auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
     assert((*off) > 0);
     int32_t ih = off - offsets - 1;
     assert(ih >= 0);
     assert(ih < nh);
     (*h).fill(v[i], i, ws, ih);
  }


  template<typename Histo, typename T>
  void fillManyFromVector(Histo * __restrict__ h, typename Histo::Counter *  __restrict__ ws,  
                          uint32_t nh, T const * __restrict__ v, uint32_t const * __restrict__ offsets, uint32_t totSize, 
                          int nthreads, cudaStream_t stream) {
    uint32_t * off = (uint32_t *)( (char*)(h) +offsetof(Histo,off));
    cudaMemsetAsync(off,0, 4*Histo::totbins(),stream);
    auto nblocks = (totSize + nthreads - 1) / nthreads;
    countFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
    cudaCheck(cudaGetLastError());
    size_t wss = Histo::totbins();
    CubDebugExit(cub::DeviceScan::InclusiveSum(ws, wss, off, off, Histo::totbins(), stream));
    cudaMemsetAsync(ws,0, 4*Histo::totbins(),stream);
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets,ws);
    cudaCheck(cudaGetLastError());
  }


} // namespace cudautils
#endif


// iteratate over N bins left and right of the one containing "v"
template<typename Hist, typename V, typename Func>
__host__ __device__
__forceinline__
void forEachInBins(Hist const & hist, V value, int n, Func func) {
   int bs = Hist::bin(value);
   int be = std::min(int(Hist::nbins()-1),bs+n);
   bs = std::max(0,bs-n);
   assert(be>=bs);
   for (auto pj=hist.begin(bs);pj<hist.end(be);++pj) {
      func(*pj);
   }
}

// iteratate over bins containing all values in window wmin, wmax
template<typename Hist, typename V, typename Func>
__host__ __device__
__forceinline__
void forEachInWindow(Hist const & hist, V wmin, V wmax, Func const & func) {
   auto bs = Hist::bin(wmin);
   auto be = Hist::bin(wmax);
   assert(be>=bs);
   for (auto pj=hist.begin(bs);pj<hist.end(be);++pj) {
      func(*pj);
   }
}



template<
  typename T, // the type of the discretized input values
  uint32_t NBINS, // number of bins 
  uint32_t SIZE, // max number of element
  uint32_t S=sizeof(T) * 8, // number of significant bits in T
  typename I=uint32_t,  // type stored in the container (usually an index in a vector of the input values)
  uint32_t NHISTS=1 // number of histos stored
>
class HistoContainer {
public:
#ifdef __CUDACC__
  using Counter = uint32_t;
#else
  using Counter = std::atomic<uint32_t>;
#endif

  using index_type = I;
  using UT = typename std::make_unsigned<T>::type;

  static constexpr uint32_t ilog2(uint32_t v) {

    constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    constexpr uint32_t s[] = {1, 2, 4, 8, 16};

    uint32_t r = 0; // result of log2(v) will go here
    for (auto i = 4; i >= 0; i--) if (v & b[i]) {
      v >>= s[i];
      r |= s[i];
    }
    return r;
  }


  static constexpr uint32_t sizeT()     { return S; }
  static constexpr uint32_t nbins()     { return NBINS;}
  static constexpr uint32_t nhists()    { return NHISTS;}
  static constexpr uint32_t totbins()   { return NHISTS*NBINS+1;}
  static constexpr uint32_t nbits()     { return ilog2(NBINS-1)+1;}
  static constexpr uint32_t capacity()  { return SIZE; }

  static constexpr auto histOff(uint32_t nh) { return NBINS*nh; }

#ifdef __CUDACC__
  __host__
  static size_t wsSize() {
    uint32_t * v =nullptr;
    void * d_temp_storage = nullptr;
    size_t  temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v, v, totbins()-1);
    return std::max(temp_storage_bytes,size_t(totbins()));
  }
#endif


  static constexpr UT bin(T t) {
    constexpr uint32_t shift = sizeT() - nbits();
    constexpr uint32_t mask = (1<<nbits()) - 1;
    return (t >> shift) & mask;
  }

  void zero() {
    for (auto & i : off)
      i = 0;
  }

  static __host__ __device__
  __forceinline__
  uint32_t atomicIncrement(Counter & x) {
    #ifdef __CUDA_ARCH__
    return atomicAdd(&x, 1);
    #else
    return x++;
    #endif
  }

  __host__ __device__
  __forceinline__
  void count(T t) {
    uint32_t b = bin(t);
    assert(b<nbins());
    atomicIncrement(off[b+1]);
  }

  __host__ __device__
  __forceinline__
  void fill(T t, index_type j, Counter * ws) {
    uint32_t b = bin(t);
    assert(b<nbins());
    auto w = atomicIncrement(ws[b]);
    assert(w < size(b));
    bins[off[b] + w] = j;
  }


  __host__ __device__
  __forceinline__
  void count(T t, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b<nbins());
    b+=histOff(nh);
    assert(b<totbins());
    atomicIncrement(off[b+1]);
  }

  __host__ __device__
  __forceinline__
  void fill(T t, index_type j, Counter * ws, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b<nbins());
    b+=histOff(nh);
    assert(b<totbins());
    auto w = atomicIncrement(ws[b]);
    assert(w < size(b));
    bins[off[b] + w] = j;
  }

#ifdef __CUDACC__
  __device__
  __forceinline__
  void finalize(Counter * ws) {
    blockPrefixScan(off+1,totbins()-1,ws);
  }
  __host__
#endif
  void finalize() {
    for(uint32_t i=2; i<totbins(); ++i) off[i]+=off[i-1];
  }

  constexpr auto size() const { return uint32_t(off[totbins()-1]);}
  constexpr auto size(uint32_t b) const { return off[b+1]-off[b];}


  constexpr index_type const * begin() const { return bins;}
  constexpr index_type const * end() const { return begin() + size();}


  constexpr index_type const * begin(uint32_t b) const { return bins + off[b];}
  constexpr index_type const * end(uint32_t b) const { return bins + off[b+1];}


  Counter  off[totbins()];
  index_type bins[capacity()];
};

#endif // HeterogeneousCore_CUDAUtilities_HistoContainer_h
