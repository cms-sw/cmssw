#include "CondFormats/EcalObjects/interface/EcalRechitADCToGeVConstantGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalRechitADCToGeVConstantGPU::EcalRechitADCToGeVConstantGPU(EcalADCToGeVConstant const& values)
    : adc2gev_(2)  // size is 2, one form EB and one for EE
{
  adc2gev_[0] = values.getEBValue();
  adc2gev_[1] = values.getEEValue();
}

EcalRechitADCToGeVConstantGPU::Product const& EcalRechitADCToGeVConstantGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalRechitADCToGeVConstantGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.adc2gev = cms::cuda::make_device_unique<float[]>(adc2gev_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.adc2gev, adc2gev_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalRechitADCToGeVConstantGPU);
