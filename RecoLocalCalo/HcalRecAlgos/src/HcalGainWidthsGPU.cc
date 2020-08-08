#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainWidthsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalGainWidthsGPU::HcalGainWidthsGPU(HcalGainWidths const& gains)
    : totalChannels_{gains.getAllContainers()[0].second.size() + gains.getAllContainers()[1].second.size()},
      value0_(totalChannels_),
      value1_(totalChannels_),
      value2_(totalChannels_),
      value3_(totalChannels_) {
  auto const gainContainers = gains.getAllContainers();

  // fill in eb
  auto const& barrelValues = gainContainers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    value0_[i] = barrelValues[i].getValue(0);
    value1_[i] = barrelValues[i].getValue(1);
    value2_[i] = barrelValues[i].getValue(2);
    value3_[i] = barrelValues[i].getValue(3);
  }

  // fill in ee
  auto const& endcapValues = gainContainers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    value0_[i + offset] = endcapValues[i].getValue(0);
    value1_[i + offset] = endcapValues[i].getValue(1);
    value2_[i + offset] = endcapValues[i].getValue(2);
    value3_[i + offset] = endcapValues[i].getValue(3);
  }
}

HcalGainWidthsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(value0));
  cudaCheck(cudaFree(value1));
  cudaCheck(cudaFree(value2));
  cudaCheck(cudaFree(value3));
}

HcalGainWidthsGPU::Product const& HcalGainWidthsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalGainWidthsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.value0, this->value0_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.value1, this->value1_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.value2, this->value2_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.value3, this->value3_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.value0,
                                  this->value0_.data(),
                                  this->value0_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.value1,
                                  this->value1_.data(),
                                  this->value1_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.value2,
                                  this->value2_.data(),
                                  this->value2_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.value3,
                                  this->value3_.data(),
                                  this->value3_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalGainWidthsGPU);
