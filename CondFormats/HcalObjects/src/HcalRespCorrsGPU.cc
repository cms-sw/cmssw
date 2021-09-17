#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

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

HcalRespCorrsGPU::Product const& HcalRespCorrsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalRespCorrsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.values, values_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalRespCorrsGPU);
