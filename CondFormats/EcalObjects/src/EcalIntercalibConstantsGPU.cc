#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalIntercalibConstantsGPU::EcalIntercalibConstantsGPU(EcalIntercalibConstants const& values) {
  values_.reserve(values.size());
  std::copy(values.begin(), values.end(), values_.begin());
  offset_ = values.barrelItems().size();
}

EcalIntercalibConstantsGPU::Product const& EcalIntercalibConstantsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalIntercalibConstantsGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.values, values_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalIntercalibConstantsGPU);
