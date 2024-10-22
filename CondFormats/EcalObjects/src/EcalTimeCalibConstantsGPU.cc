#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstantsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalTimeCalibConstantsGPU::EcalTimeCalibConstantsGPU(EcalTimeCalibConstants const& values) {
  values_.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values_.emplace_back(values[i]);
  }
  offset_ = values.barrelItems().size();
}

EcalTimeCalibConstantsGPU::Product const& EcalTimeCalibConstantsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalTimeCalibConstantsGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.values, values_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalTimeCalibConstantsGPU);
