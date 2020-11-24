#ifndef CondFormats_HcalObjects_interface_HcalQIETypesGPU_h
#define CondFormats_HcalObjects_interface_HcalQIETypesGPU_h

#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalQIETypesGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> values;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalQIETypesGPU(HcalQIETypes const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalQIETypesGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  std::vector<int, cms::cuda::HostAllocator<int>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
