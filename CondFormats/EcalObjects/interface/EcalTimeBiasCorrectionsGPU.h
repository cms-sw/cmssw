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
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> ebTimeCorrAmplitudeBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> ebTimeCorrShiftBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> eeTimeCorrAmplitudeBins;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> eeTimeCorrShiftBins;
    int ebTimeCorrAmplitudeBinsSize, eeTimeCorrAmplitudeBinsSize;
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
  std::vector<float, cms::cuda::HostAllocator<float>> ebTimeCorrAmplitudeBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> ebTimeCorrShiftBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> eeTimeCorrAmplitudeBins_;
  std::vector<float, cms::cuda::HostAllocator<float>> eeTimeCorrShiftBins_;

#ifndef __CUDACC__
  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // CondFormats_EcalObjects_interface_EcalTimeBiasCorrectionsGPU_h
