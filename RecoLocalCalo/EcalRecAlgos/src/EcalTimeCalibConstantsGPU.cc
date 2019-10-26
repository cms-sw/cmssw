#include "RecoLocalCalo/EcalRecAlgos/interface/EcalTimeCalibConstantsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalTimeCalibConstantsGPU::EcalTimeCalibConstantsGPU(EcalTimeCalibConstants const& values)
    : valuesEB_{values.barrelItems()}, valuesEE_{values.endcapItems()} {}

EcalTimeCalibConstantsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

EcalTimeCalibConstantsGPU::Product const& EcalTimeCalibConstantsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalTimeCalibConstantsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(
            cudaMalloc((void**)&product.values, (this->valuesEB_.size() + this->valuesEE_.size()) * sizeof(float)));

        // offset in floats, not bytes
        auto const offset = this->valuesEB_.size();

        // transfer
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->valuesEB_.data(),
                                  this->valuesEB_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.values + offset,
                                  this->valuesEE_.data(),
                                  this->valuesEE_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalTimeCalibConstantsGPU);
