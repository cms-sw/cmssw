#ifndef CondFormats_HcalObjects_interface_HcalGainWidthsGPU_h
#define CondFormats_HcalObjects_interface_HcalGainWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalGainWidthsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value0;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value2;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value3;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalGainWidthsGPU(HcalGainWidths const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalGainWidthsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> value0_, value1_, value2_, value3_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
