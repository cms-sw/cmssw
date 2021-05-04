#ifndef CondFormats_HcalObjects_interface_HcalRespCorrsGPU_h
#define CondFormats_HcalObjects_interface_HcalRespCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalRespCorrsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> values;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalRespCorrsGPU(HcalRespCorrs const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalRespCorrsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<float, cms::cuda::HostAllocator<float>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
