#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIETypesGPU.h"

#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalQIETypesGPU::HcalQIETypesGPU(HcalQIETypes const& parameters)
    : values_(parameters.getAllContainers()[0].second.size() + parameters.getAllContainers()[1].second.size()) {
  auto const containers = parameters.getAllContainers();

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

HcalQIETypesGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

HcalQIETypesGPU::Product const& HcalQIETypesGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalQIETypesGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->values_.data(),
                                  this->values_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalQIETypesGPU);
