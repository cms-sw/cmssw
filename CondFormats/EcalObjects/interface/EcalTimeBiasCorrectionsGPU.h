#ifndef CondFormats_EcalObjects_interface_EcalTimeBiasCorrectionsGPU_h
#define CondFormats_EcalObjects_interface_EcalTimeBiasCorrectionsGPU_h

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalTimeBiasCorrectionsGPU {
public:
  struct Product {
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> EBTimeCorrAmplitudeBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> EBTimeCorrShiftBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> EETimeCorrAmplitudeBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> EETimeCorrShiftBins;
    int EBTimeCorrAmplitudeBinsSize, EETimeCorrAmplitudeBinsSize;
  };

  // rearrange pedestals
  EcalTimeBiasCorrectionsGPU(EcalTimeBiasCorrections const&);

#ifndef __CUDACC__

  // will call dealloation for Product thru ~Product
  ~EcalTimeBiasCorrectionsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalTimeBiasCorrectionsGPU"}; }
#endif  // __CUDACC__

private:
  std::vector<float, cms::cuda::HostAllocator<float>> EBTimeCorrAmplitudeBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> EBTimeCorrShiftBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> EETimeCorrAmplitudeBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> EETimeCorrShiftBins_;

#ifndef __CUDACC__
  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalTimeBiasCorrectionsGPU_h
