#ifndef CondFormats_HcalObjects_interface_HcalPedestalsGPU_h
#define CondFormats_HcalObjects_interface_HcalPedestalsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalPedestalsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> values;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> widths;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalPedestalsGPU(HcalPedestals const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalPedestalsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

  // as in cpu version
  bool unitIsADC() const { return unitIsADC_; }

  uint32_t offsetForHashes() const { return offsetForHashes_; }

private:
  bool unitIsADC_;
  uint64_t totalChannels_;
  uint32_t offsetForHashes_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_;
  std::vector<float, cms::cuda::HostAllocator<float>> widths_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
