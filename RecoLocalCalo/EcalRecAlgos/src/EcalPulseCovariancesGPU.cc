#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseCovariancesGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalPulseCovariancesGPU::EcalPulseCovariancesGPU(EcalPulseCovariances const& values)
    : valuesEB_{values.barrelItems()}, valuesEE_{values.endcapItems()} {}

EcalPulseCovariancesGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

EcalPulseCovariancesGPU::Product const& EcalPulseCovariancesGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalPulseCovariancesGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values,
                             (this->valuesEE_.size() + this->valuesEB_.size()) * sizeof(EcalPulseCovariance)));

        // offset in terms of sizeof(EcalPulseCovariance)
        uint32_t offset = this->valuesEB_.size();

        // transfer eb
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->valuesEB_.data(),
                                  this->valuesEB_.size() * sizeof(EcalPulseCovariance),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        // transfer ee starting at values + offset
        cudaCheck(cudaMemcpyAsync(product.values + offset,
                                  this->valuesEE_.data(),
                                  this->valuesEE_.size() * sizeof(EcalPulseCovariance),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalPulseCovariancesGPU);
