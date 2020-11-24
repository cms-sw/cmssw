#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypesGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

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

HcalQIETypesGPU::Product const& HcalQIETypesGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalQIETypesGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.values = cms::cuda::make_device_unique<int[]>(values_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.values, values_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalQIETypesGPU);
