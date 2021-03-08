#include "CondFormats/EcalObjects/interface/EcalLaserAlphasGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalLaserAlphasGPU::EcalLaserAlphasGPU(EcalLaserAlphas const& values) {
  values_.reserve(values.size());
  std::copy(values.begin(), values.end(), values_.begin());
  offset_ = values.barrelItems().size();
}

EcalLaserAlphasGPU::Product const& EcalLaserAlphasGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalLaserAlphasGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.values, values_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalLaserAlphasGPU);
