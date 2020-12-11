#ifndef CondFormats_HcalObjects_interface_HcalLUTCorrsGPU_h
#define CondFormats_HcalObjects_interface_HcalLUTCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalLUTCorrsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalLUTCorrsGPU(HcalLUTCorrs const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalLUTCorrsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<float, cms::cuda::HostAllocator<float>> value_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
