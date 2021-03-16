#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrectionsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalTimeBiasCorrectionsGPU::EcalTimeBiasCorrectionsGPU(EcalTimeBiasCorrections const& values) {
  ebTimeCorrAmplitudeBins_.reserve(values.EBTimeCorrAmplitudeBins.size());
  for (const auto& ebTimeCorrAmplitudeBin : values.EBTimeCorrAmplitudeBins) {
    ebTimeCorrAmplitudeBins_.emplace_back(ebTimeCorrAmplitudeBin);
  }

  ebTimeCorrShiftBins_.reserve(values.EBTimeCorrAmplitudeBins.size());
  for (const auto& ebTimeCorrShiftBin : values.EBTimeCorrShiftBins) {
    ebTimeCorrShiftBins_.emplace_back(ebTimeCorrShiftBin);
  }

  eeTimeCorrAmplitudeBins_.reserve(values.EETimeCorrAmplitudeBins.size());
  for (const auto& eeTimeCorrAmplitudeBin : values.EETimeCorrAmplitudeBins) {
    eeTimeCorrAmplitudeBins_.emplace_back(eeTimeCorrAmplitudeBin);
  }

  eeTimeCorrShiftBins_.reserve(values.EETimeCorrAmplitudeBins.size());
  for (const auto& eeTimeCorrShiftBin : values.EETimeCorrShiftBins) {
    eeTimeCorrShiftBins_.emplace_back(eeTimeCorrShiftBin);
  }
}

EcalTimeBiasCorrectionsGPU::Product const& EcalTimeBiasCorrectionsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalTimeBiasCorrectionsGPU::Product& product, cudaStream_t cudaStream) {
        // to get the size of vectors later on
        // should be removed and host conditions' objects used directly
        product.ebTimeCorrAmplitudeBinsSize = this->ebTimeCorrAmplitudeBins_.size();
        product.eeTimeCorrAmplitudeBinsSize = this->eeTimeCorrAmplitudeBins_.size();

        // allocate
        product.ebTimeCorrAmplitudeBins =
            cms::cuda::make_device_unique<float[]>(ebTimeCorrAmplitudeBins_.size(), cudaStream);
        product.ebTimeCorrShiftBins = cms::cuda::make_device_unique<float[]>(ebTimeCorrShiftBins_.size(), cudaStream);
        product.eeTimeCorrAmplitudeBins =
            cms::cuda::make_device_unique<float[]>(eeTimeCorrAmplitudeBins_.size(), cudaStream);
        product.eeTimeCorrShiftBins = cms::cuda::make_device_unique<float[]>(eeTimeCorrShiftBins_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.ebTimeCorrAmplitudeBins, ebTimeCorrAmplitudeBins_, cudaStream);
        cms::cuda::copyAsync(product.ebTimeCorrShiftBins, ebTimeCorrShiftBins_, cudaStream);
        cms::cuda::copyAsync(product.eeTimeCorrAmplitudeBins, eeTimeCorrAmplitudeBins_, cudaStream);
        cms::cuda::copyAsync(product.eeTimeCorrShiftBins, eeTimeCorrShiftBins_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalTimeBiasCorrectionsGPU);
