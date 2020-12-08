#ifndef CondFormats_HcalObjects_interface_HcalQIECodersGPU_h
#define CondFormats_HcalObjects_interface_HcalQIECodersGPU_h

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalQIECodersGPU {
public:
  static constexpr uint32_t numValuesPerChannel = 16;

  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> offsets;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> slopes;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalQIECodersGPU(HcalQIEData const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalQIECodersGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> offsets_;
  std::vector<float, cms::cuda::HostAllocator<float>> slopes_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
