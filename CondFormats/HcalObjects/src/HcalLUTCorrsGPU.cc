#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

// FIXME: add proper getters to conditions
HcalLUTCorrsGPU::HcalLUTCorrsGPU(HcalLUTCorrs const& lutcorrs)
    : value_(lutcorrs.getAllContainers()[0].second.size() + lutcorrs.getAllContainers()[1].second.size()) {
  auto const containers = lutcorrs.getAllContainers();

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

HcalLUTCorrsGPU::Product const& HcalLUTCorrsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalLUTCorrsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.value = cms::cuda::make_device_unique<float[]>(value_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.value, value_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalLUTCorrsGPU);
