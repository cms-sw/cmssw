#ifndef CondFormats_HcalObjects_interface_HcalConvertedPedestalWidthsGPU_h
#define CondFormats_HcalObjects_interface_HcalConvertedPedestalWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalConvertedPedestalWidthsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> values;
  };

#ifndef __CUDACC__
  // order matters!
  HcalConvertedPedestalWidthsGPU(HcalPedestals const&,
                                 HcalPedestalWidths const&,
                                 HcalQIEData const&,
                                 HcalQIETypes const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalConvertedPedestalWidthsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
