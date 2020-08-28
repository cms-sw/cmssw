#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalGains.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalGainsGPU::HcalGainsGPU(HcalGains const& gains)
    : totalChannels_{gains.getAllContainers()[0].second.size() + gains.getAllContainers()[1].second.size()},
      values_(totalChannels_ * 4) {
  auto const gainContainers = gains.getAllContainers();

  // fill in eb
  auto const& barrelValues = gainContainers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    values_[i * 4] = barrelValues[i].getValue(0);
    values_[i * 4 + 1] = barrelValues[i].getValue(1);
    values_[i * 4 + 2] = barrelValues[i].getValue(2);
    values_[i * 4 + 3] = barrelValues[i].getValue(3);
  }

  // fill in ee
  auto const& endcapValues = gainContainers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    auto const off = offset + i;
    values_[off * 4] = endcapValues[i].getValue(0);
    values_[off * 4 + 1] = endcapValues[i].getValue(1);
    values_[off * 4 + 2] = endcapValues[i].getValue(2);
    values_[off * 4 + 3] = endcapValues[i].getValue(3);
  }
}

HcalGainsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

HcalGainsGPU::Product const& HcalGainsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(cudaStream, [this](HcalGainsGPU::Product& product, cudaStream_t cudaStream) {
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

TYPELOOKUP_DATA_REG(HcalGainsGPU);
