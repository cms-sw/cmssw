#include "RecoLocalCalo/EcalRecAlgos/interface/EcalLaserAPDPNRatiosRefGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalLaserAPDPNRatiosRefGPU::EcalLaserAPDPNRatiosRefGPU(EcalLaserAPDPNRatiosRef const& values)
    : valuesEB_{values.barrelItems()}, valuesEE_{values.endcapItems()} {}

EcalLaserAPDPNRatiosRefGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

EcalLaserAPDPNRatiosRefGPU::Product const& EcalLaserAPDPNRatiosRefGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalLaserAPDPNRatiosRefGPU::Product& product, cudaStream_t cudaStream) {
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

TYPELOOKUP_DATA_REG(EcalLaserAPDPNRatiosRefGPU);
