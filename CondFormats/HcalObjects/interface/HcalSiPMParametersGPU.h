#ifndef CondFormats_HcalObjects_interface_HcalSiPMParametersGPU_h
#define CondFormats_HcalObjects_interface_HcalSiPMParametersGPU_h

#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalSiPMParametersGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> type;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> auxi1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> fcByPE;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> darkCurrent;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> auxi2;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalSiPMParametersGPU(HcalSiPMParameters const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalSiPMParametersGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<int, cms::cuda::HostAllocator<int>> type_, auxi1_;
  std::vector<float, cms::cuda::HostAllocator<float>> fcByPE_, darkCurrent_, auxi2_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalSiPMParametersGPU_h
