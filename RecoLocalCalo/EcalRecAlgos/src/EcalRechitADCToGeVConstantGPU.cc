#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitADCToGeVConstantGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalRechitADCToGeVConstantGPU::EcalRechitADCToGeVConstantGPU(EcalADCToGeVConstant const& values)
    : adc2gev_(2)  // size is 2, one form EB and one for EE
{
  adc2gev_[0] = values.getEBValue();
  adc2gev_[1] = values.getEEValue();
}

EcalRechitADCToGeVConstantGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(adc2gev));
}

EcalRechitADCToGeVConstantGPU::Product const& EcalRechitADCToGeVConstantGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalRechitADCToGeVConstantGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.adc2gev, this->adc2gev_.size() * sizeof(float)));
        // transfer
        cudaCheck(cudaMemcpyAsync(product.adc2gev,
                                  this->adc2gev_.data(),
                                  this->adc2gev_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalRechitADCToGeVConstantGPU);
