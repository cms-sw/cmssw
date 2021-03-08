#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelationGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalSamplesCorrelationGPU::EcalSamplesCorrelationGPU(EcalSamplesCorrelation const& values) {
  EBG12SamplesCorrelation_.reserve(values.EBG12SamplesCorrelation.size());
  for (const auto& EBG12SamplesCorrelation : values.EBG12SamplesCorrelation) {
    EBG12SamplesCorrelation_.emplace_back(EBG12SamplesCorrelation);
  }

  EBG6SamplesCorrelation_.reserve(values.EBG6SamplesCorrelation.size());
  for (const auto& EBG6SamplesCorrelation : values.EBG6SamplesCorrelation) {
    EBG6SamplesCorrelation_.emplace_back(EBG6SamplesCorrelation);
  }

  EBG1SamplesCorrelation_.reserve(values.EBG1SamplesCorrelation.size());
  for (const auto& EBG1SamplesCorrelation : values.EBG1SamplesCorrelation) {
    EBG1SamplesCorrelation_.emplace_back(EBG1SamplesCorrelation);
  }

  EEG12SamplesCorrelation_.reserve(values.EEG12SamplesCorrelation.size());
  for (const auto& EEG12SamplesCorrelation : values.EEG12SamplesCorrelation) {
    EEG12SamplesCorrelation_.emplace_back(EEG12SamplesCorrelation);
  }

  EEG6SamplesCorrelation_.reserve(values.EEG6SamplesCorrelation.size());
  for (const auto& EEG6SamplesCorrelation : values.EEG6SamplesCorrelation) {
    EEG6SamplesCorrelation_.emplace_back(EEG6SamplesCorrelation);
  }

  EEG1SamplesCorrelation_.reserve(values.EEG1SamplesCorrelation.size());
  for (const auto& EEG1SamplesCorrelation : values.EEG1SamplesCorrelation) {
    EEG1SamplesCorrelation_.emplace_back(EEG1SamplesCorrelation);
  }
}

EcalSamplesCorrelationGPU::Product const& EcalSamplesCorrelationGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalSamplesCorrelationGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.EBG12SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EBG12SamplesCorrelation_.size(), cudaStream);
        product.EBG6SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EBG6SamplesCorrelation_.size(), cudaStream);
        product.EBG1SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EBG1SamplesCorrelation_.size(), cudaStream);
        product.EEG12SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EEG12SamplesCorrelation_.size(), cudaStream);
        product.EEG6SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EEG6SamplesCorrelation_.size(), cudaStream);
        product.EEG1SamplesCorrelation =
            cms::cuda::make_device_unique<double[]>(EEG1SamplesCorrelation_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.EBG12SamplesCorrelation, EBG12SamplesCorrelation_, cudaStream);
        cms::cuda::copyAsync(product.EBG6SamplesCorrelation, EBG6SamplesCorrelation_, cudaStream);
        cms::cuda::copyAsync(product.EBG1SamplesCorrelation, EBG1SamplesCorrelation_, cudaStream);
        cms::cuda::copyAsync(product.EEG12SamplesCorrelation, EEG12SamplesCorrelation_, cudaStream);
        cms::cuda::copyAsync(product.EEG6SamplesCorrelation, EEG6SamplesCorrelation_, cudaStream);
        cms::cuda::copyAsync(product.EEG1SamplesCorrelation, EEG1SamplesCorrelation_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalSamplesCorrelationGPU);
