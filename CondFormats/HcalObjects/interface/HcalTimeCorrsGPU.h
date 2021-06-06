#ifndef CondFormats_HcalObjects_interface_HcalTimeCorrsGPU_h
#define CondFormats_HcalObjects_interface_HcalTimeCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalTimeCorrsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> value;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalTimeCorrsGPU(HcalTimeCorrs const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalTimeCorrsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<float, cms::cuda::HostAllocator<float>> value_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
