#ifndef HeterogeneousCore_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers
#define HeterogeneousCore_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers

#include "cuda/api_wrappers.h"

#include <functional>
#include <memory>
#include <utility>

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input);

class TestAcceleratorServiceProducerGPUTask {
public:
  TestAcceleratorServiceProducerGPUTask();
  ~TestAcceleratorServiceProducerGPUTask() = default;

  using Ptr = cuda::memory::device::unique_ptr<float[]>;
  using PtrRaw = Ptr::pointer;
  
  using ResultType = std::pair<Ptr, Ptr>;
  using ResultTypeRaw = std::pair<PtrRaw, PtrRaw>;
  using ConstResultTypeRaw = std::pair<const PtrRaw, const PtrRaw>;

  ResultType runAlgo(int input, const ResultTypeRaw inputArrays, std::function<void()> callback);
  void release();
  int getResult(const ResultTypeRaw& d_ac);

private:
  std::unique_ptr<cuda::stream_t<>> streamPtr;

  // temporary storage, need to be somewhere to allow async execution
  cuda::memory::host::unique_ptr<float[]> h_a;
  cuda::memory::host::unique_ptr<float[]> h_b;
  cuda::memory::device::unique_ptr<float[]> d_b;
  cuda::memory::device::unique_ptr<float[]> d_d;
  cuda::memory::device::unique_ptr<float[]> d_ma;
  cuda::memory::device::unique_ptr<float[]> d_mb;
  cuda::memory::device::unique_ptr<float[]> d_mc;
};

#endif
