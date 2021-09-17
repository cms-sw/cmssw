#ifndef CondFormats_HcalObjects_interface_HcalPedestalWidthsGPU_h
#define CondFormats_HcalObjects_interface_HcalPedestalWidthsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalPedestalWidthsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma00;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma01;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma02;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma03;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma10;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma11;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma12;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma13;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma20;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma21;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma22;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma23;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma30;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma31;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma32;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> sigma33;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalPedestalWidthsGPU(HcalPedestalWidths const&);

  // will trigger deallocation of Product thru ~Product
  ~HcalPedestalWidthsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  // as in cpu version
  bool unitIsADC() const { return unitIsADC_; }

private:
  bool unitIsADC_;
  uint64_t totalChannels_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma00_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma01_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma02_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma03_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma10_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma11_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma12_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma13_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma20_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma21_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma22_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma23_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma30_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma31_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma32_;
  std::vector<float, cms::cuda::HostAllocator<float>> sigma33_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
