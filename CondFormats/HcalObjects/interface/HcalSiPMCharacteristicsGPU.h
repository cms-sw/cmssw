#ifndef CondFormats_HcalObjects_interface_HcalSiPMCharacteristicsGPU_h
#define CondFormats_HcalObjects_interface_HcalSiPMCharacteristicsGPU_h

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalSiPMCharacteristicsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> pixels;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> parLin1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> parLin2;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> parLin3;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> crossTalk;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<int[]>> auxi1;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> auxi2;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalSiPMCharacteristicsGPU(HcalSiPMCharacteristics const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalSiPMCharacteristicsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  std::vector<int, cms::cuda::HostAllocator<int>> pixels_, auxi1_;
  std::vector<float, cms::cuda::HostAllocator<float>> parLin1_, parLin2_, parLin3_, crossTalk_, auxi2_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
