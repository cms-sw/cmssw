#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalTimeBiasCorrectionsGPU::EcalTimeBiasCorrectionsGPU(EcalTimeBiasCorrections const& values) {
  EBTimeCorrAmplitudeBins_.reserve(values.EBTimeCorrAmplitudeBins.size());
  for (const auto& EBTimeCorrAmplitudeBin : values.EBTimeCorrAmplitudeBins) {
    EBTimeCorrAmplitudeBins_.emplace_back(EBTimeCorrAmplitudeBin);
  }

  EBTimeCorrShiftBins_.reserve(values.EBTimeCorrAmplitudeBins.size());
  for (const auto& EBTimeCorrShiftBin : values.EBTimeCorrShiftBins) {
    EBTimeCorrShiftBins_.emplace_back(EBTimeCorrShiftBin);
  }

  EETimeCorrAmplitudeBins_.reserve(values.EETimeCorrAmplitudeBins.size());
  for (const auto& EETimeCorrAmplitudeBin : values.EETimeCorrAmplitudeBins) {
    EETimeCorrAmplitudeBins_.emplace_back(EETimeCorrAmplitudeBin);
  }

  EETimeCorrShiftBins_.reserve(values.EETimeCorrAmplitudeBins.size());
  for (const auto& EETimeCorrShiftBin : values.EETimeCorrShiftBins) {
    EETimeCorrShiftBins_.emplace_back(EETimeCorrShiftBin);
  }
}

EcalTimeBiasCorrectionsGPU::Product const& EcalTimeBiasCorrectionsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalTimeBiasCorrectionsGPU::Product& product, cudaStream_t cudaStream) {
        // to get the size of vectors later on
        // should be removed and host conditions' objects used directly
        product.EBTimeCorrAmplitudeBinsSize = this->EBTimeCorrAmplitudeBins_.size();
        product.EETimeCorrAmplitudeBinsSize = this->EETimeCorrAmplitudeBins_.size();

        // allocate
        product.EBTimeCorrAmplitudeBins = cms::cuda::make_device_unique<float[]>(EBTimeCorrAmplitudeBins_.size(), cudaStream);
        product.EBTimeCorrShiftBins = cms::cuda::make_device_unique<float[]>(EBTimeCorrShiftBins_.size(), cudaStream);
        product.EETimeCorrAmplitudeBins = cms::cuda::make_device_unique<float[]>(EETimeCorrAmplitudeBins_.size(), cudaStream);
        product.EETimeCorrShiftBins = cms::cuda::make_device_unique<float[]>(EETimeCorrShiftBins_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.EBTimeCorrAmplitudeBins, EBTimeCorrAmplitudeBins_, cudaStream);
        cms::cuda::copyAsync(product.EBTimeCorrShiftBins, EBTimeCorrShiftBins_, cudaStream);
        cms::cuda::copyAsync(product.EETimeCorrAmplitudeBins, EETimeCorrAmplitudeBins_, cudaStream);
        cms::cuda::copyAsync(product.EETimeCorrShiftBins, EETimeCorrShiftBins_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalTimeBiasCorrectionsGPU);
