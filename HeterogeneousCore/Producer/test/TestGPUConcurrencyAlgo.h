#ifndef HeterogeneousCore_Producer_test_TestGPUConcurrencyAlgo_h
#define HeterogeneousCore_Producer_test_TestGPUConcurrencyAlgo_h

#include <cuda_runtime.h>

class TestGPUConcurrencyAlgo {
public:
  TestGPUConcurrencyAlgo(unsigned int blocks, unsigned int threads, unsigned int sleep) :
    blocks_(blocks),
    threads_(threads),
    sleep_(sleep)
  { }

  void kernelWrapper(cudaStream_t stream) const;

// data members
private:
  unsigned int blocks_;
  unsigned int threads_;
  unsigned int sleep_;
};

#endif // HeterogeneousCore_Producer_test_TestGPUConcurrencyAlgo_h
