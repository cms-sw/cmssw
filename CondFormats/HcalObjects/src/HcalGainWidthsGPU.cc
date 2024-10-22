#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidthsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

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

HcalGainWidthsGPU::Product const& HcalGainWidthsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalGainWidthsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.value0 = cms::cuda::make_device_unique<float[]>(value0_.size(), stream);
        product.value1 = cms::cuda::make_device_unique<float[]>(value1_.size(), stream);
        product.value2 = cms::cuda::make_device_unique<float[]>(value2_.size(), stream);
        product.value3 = cms::cuda::make_device_unique<float[]>(value3_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.value0, value0_, stream);
        cms::cuda::copyAsync(product.value1, value1_, stream);
        cms::cuda::copyAsync(product.value2, value2_, stream);
        cms::cuda::copyAsync(product.value3, value3_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalGainWidthsGPU);
