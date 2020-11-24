#ifndef CondFormats_HcalObjects_interface_HcalRecoParamsGPU_h
#define CondFormats_HcalObjects_interface_HcalRecoParamsGPU_h

#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalRecoParams;

class HcalRecoParamsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> param1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> param2;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalRecoParamsGPU(HcalRecoParams const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalRecoParamsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;  // hb + he
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> param1_;
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> param2_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
