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
#include "HeterogeneousCore/CUDAUtilities/interface/AtomicPairCounter.h"


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
  void fillFromVector(Histo * __restrict__ h,  uint32_t nh, T const * __restrict__ v, uint32_t const * __restrict__ offsets) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= offsets[nh]) return;
     auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
     assert((*off) > 0);
     int32_t ih = off - offsets - 1;
     assert(ih >= 0);
     assert(ih < nh);
     (*h).fill(v[i], i, ih);
  }

  template<typename Histo>
  void launchZero(Histo * __restrict__ h, cudaStream_t stream) {
    uint32_t * off = (uint32_t *)( (char*)(h) +offsetof(Histo,off));
    cudaMemsetAsync(off,0, 4*Histo::totbins(),stream);
  }

  template<typename Histo>
  void launchFinalize(Histo * __restrict__ h, uint8_t *  __restrict__ ws, cudaStream_t stream) {
    uint32_t * off = (uint32_t *)( (char*)(h) +offsetof(Histo,off));
    size_t wss = Histo::wsSize();
    CubDebugExit(cub::DeviceScan::InclusiveSum(ws, wss, off, off, Histo::totbins(), stream));
  }


  template<typename Histo, typename T>
  void fillManyFromVector(Histo * __restrict__ h, uint8_t *  __restrict__ ws,  
                          uint32_t nh, T const * __restrict__ v, uint32_t const * __restrict__ offsets, uint32_t totSize, 
                          int nthreads, cudaStream_t stream) {
    launchZero(h,stream); 
    auto nblocks = (totSize + nthreads - 1) / nthreads;
    countFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
    cudaCheck(cudaGetLastError());
    launchFinalize(h,ws,stream);
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
    cudaCheck(cudaGetLastError());
  }


  template<typename Assoc>
  __global__
  void finalizeBulk(AtomicPairCounter const * apc, Assoc * __restrict__ assoc) {
     assoc->bulkFinalizeFill(*apc);
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
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, v, v, totbins());
    return temp_storage_bytes;
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

  static __host__ __device__
  __forceinline__
  uint32_t atomicDecrement(Counter & x) {
    #ifdef __CUDA_ARCH__
    return atomicSub(&x, 1);
    #else
    return x--;
    #endif
  }

 __host__ __device__
  __forceinline__
  void countDirect(T b) {
    assert(b<nbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__
  __forceinline__
  void fillDirect(T b, index_type j) {
    assert(b<nbins());
    auto w = atomicDecrement(off[b]);
    assert(w>0);
    bins[w-1] = j;
  }


#ifdef __CUDACC__
  __device__
  __forceinline__
  uint32_t bulkFill(AtomicPairCounter & apc, index_type const * v, uint32_t n) {
    auto c = apc.add(n);
    off[c.m] = c.n;
    for(int j=0; j<n; ++j) bins[c.n+j]=v[j];
    return c.m;
  }

  __device__
  __forceinline__
  void bulkFinalize(AtomicPairCounter const & apc) {
     off[apc.get().m]=apc.get().n;
  }

  __device__
  __forceinline__
  void bulkFinalizeFill(AtomicPairCounter const & apc) {
     auto m = apc.get().m;
     auto n = apc.get().n;
     auto i = m + blockIdx.x * blockDim.x + threadIdx.x;
     if (i>=totbins()) return;
     off[i]=n;
  }


#endif


  __host__ __device__
  __forceinline__
  void count(T t) {
    uint32_t b = bin(t);
    assert(b<nbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__
  __forceinline__
  void fill(T t, index_type j) {
    uint32_t b = bin(t);
    assert(b<nbins());
    auto w = atomicDecrement(off[b]);
    assert(w>0);
    bins[w-1] = j;
  }


  __host__ __device__
  __forceinline__
  void count(T t, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b<nbins());
    b+=histOff(nh);
    assert(b<totbins());
    atomicIncrement(off[b]);
  }

  __host__ __device__
  __forceinline__
  void fill(T t, index_type j, uint32_t nh) {
    uint32_t b = bin(t);
    assert(b<nbins());
    b+=histOff(nh);
    assert(b<totbins());
    auto w = atomicDecrement(off[b]);
    assert(w>0);
    bins[w-1] = j;
  }

#ifdef __CUDACC__
  __device__
  __forceinline__
  void finalize(Counter * ws) {
    assert(off[totbins()-1]==0);
    blockPrefixScan(off,totbins(),ws);
    assert(off[totbins()-1]==off[totbins()-2]);
  }
  __host__
#endif
  void finalize() {
    assert(off[totbins()-1]==0);
    for(uint32_t i=1; i<totbins(); ++i) off[i]+=off[i-1];
    assert(off[totbins()-1]==off[totbins()-2]);
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



template<
  typename I, // type stored in the container (usually an index in a vector of the input values)
  uint32_t MAXONES, // max number of "ones"
  uint32_t MAXMANYS // max number of "manys"
>
using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

#endif // HeterogeneousCore_CUDAUtilities_HistoContainer_h
