#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

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

HcalTimeCorrsGPU::Product const& HcalTimeCorrsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalTimeCorrsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.value = cms::cuda::make_device_unique<float[]>(value_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.value, value_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalTimeCorrsGPU);
