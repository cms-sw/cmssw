#ifndef HeterogeneousCoreCUDAUtilitiesAtomicPairCounter_H
#define HeterogeneousCoreCUDAUtilitiesAtomicPairCounter_H

#include <cuda_runtime.h>
#include <cstdint>

class AtomicPairCounter {
public:

  using c_type = unsigned long long int;

  AtomicPairCounter(){}
  AtomicPairCounter(c_type i) { counter.ac=i;}

  __device__ __host__
  AtomicPairCounter & operator=(c_type i) { counter.ac=i; return *this;}

  struct Counters {
    uint32_t n;  // in a "One to Many" association is the number of "One"
    uint32_t m;  // in a "One to Many" association is the total number of associations
  };

  union Atomic2 {
    Counters counters;
    c_type ac;
  };

#ifdef __CUDACC__

  static constexpr c_type incr = 1UL<<32;

  __device__ __host__
  Counters get() const { return counter.counters;}

  // increment n by 1 and m by i.  return previous value
  __device__
  Counters add(uint32_t i) {
    c_type c = i; 
    c+=incr;
    Atomic2 ret;
    ret.ac = atomicAdd(&counter.ac,c);
    return ret.counters;
  }

#endif

private:

  Atomic2 counter;

};


#endif
