#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalTimeBiasCorrectionsGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalTimeBiasCorrectionsGPU_h

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalTimeBiasCorrectionsGPU {
public:
  struct Product {
    ~Product();
    float *EBTimeCorrAmplitudeBins, *EBTimeCorrShiftBins;
    float *EETimeCorrAmplitudeBins, *EETimeCorrShiftBins;
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

  std::vector<float> const& EBTimeCorrAmplitudeBins() const { return EBTimeCorrAmplitudeBins_; }
  std::vector<float> const& EETimeCorrAmplitudeBins() const { return EETimeCorrAmplitudeBins_; }

private:
  std::vector<float> const& EBTimeCorrAmplitudeBins_;
  std::vector<float> const& EBTimeCorrShiftBins_;
  std::vector<float> const& EETimeCorrAmplitudeBins_;
  std::vector<float> const& EETimeCorrShiftBins_;

#ifndef __CUDACC__
  cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalTimeBiasCorrectionsGPU_h
