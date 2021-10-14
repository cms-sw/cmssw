#ifndef CondFormats_HcalObjects_interface_HcalChannelQualityGPU_h
#define CondFormats_HcalObjects_interface_HcalChannelQualityGPU_h

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalChannelQualityGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> status;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalChannelQualityGPU(HcalChannelQuality const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalChannelQualityGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> status_;

  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalChannelQualityGPU_h
