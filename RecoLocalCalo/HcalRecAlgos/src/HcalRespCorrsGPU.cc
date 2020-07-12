#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalRespCorrsGPU::HcalRespCorrsGPU(HcalRespCorrs const& respcorrs)
    : values_(respcorrs.getAllContainers()[0].second.size() + respcorrs.getAllContainers()[1].second.size()) {
  auto const containers = respcorrs.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    values_[i] = barrelValues[i].getValue();
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    values_[i + offset] = endcapValues[i].getValue();
  }
}

HcalRespCorrsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

HcalRespCorrsGPU::Product const& HcalRespCorrsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalRespCorrsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->values_.data(),
                                  this->values_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalRespCorrsGPU);
