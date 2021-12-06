#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQualityGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

// FIXME: add proper getters to conditions
HcalChannelQualityGPU::HcalChannelQualityGPU(HcalChannelQuality const& quality)
    : totalChannels_{quality.getAllContainers()[0].second.size() + quality.getAllContainers()[1].second.size()},
      status_(totalChannels_) {
  auto const containers = quality.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    status_[i] = barrelValues[i].getValue();
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    status_[i + offset] = endcapValues[i].getValue();
  }
}

HcalChannelQualityGPU::Product const& HcalChannelQualityGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalChannelQualityGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.status = cms::cuda::make_device_unique<uint32_t[]>(status_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.status, status_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalChannelQualityGPU);
