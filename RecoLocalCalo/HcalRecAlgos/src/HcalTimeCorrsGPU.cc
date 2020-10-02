#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimeCorrsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalTimeCorrsGPU::HcalTimeCorrsGPU(HcalTimeCorrs const& timecorrs)
    : value_(timecorrs.getAllContainers()[0].second.size() + timecorrs.getAllContainers()[1].second.size()) {
  auto const containers = timecorrs.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    value_[i] = barrelValues[i].getValue();
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    value_[i + offset] = endcapValues[i].getValue();
  }
}

HcalTimeCorrsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(value));
}

HcalTimeCorrsGPU::Product const& HcalTimeCorrsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalTimeCorrsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.value, this->value_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.value,
                                  this->value_.data(),
                                  this->value_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalTimeCorrsGPU);
