#ifndef CondFormats_HcalObjects_interface_HcalConvertedPedestalsGPU_h
#define CondFormats_HcalObjects_interface_HcalConvertedPedestalsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalConvertedPedestalsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> values;
  };

#ifndef __CUDACC__
  // order matters!
  HcalConvertedPedestalsGPU(HcalPedestals const&, HcalQIEData const&, HcalQIETypes const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalConvertedPedestalsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  uint32_t offsetForHashes() const { return offsetForHashes_; }

protected:
  uint64_t totalChannels_;
  uint32_t offsetForHashes_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
