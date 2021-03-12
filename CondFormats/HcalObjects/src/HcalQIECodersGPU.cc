#include "CondFormats/HcalObjects/interface/HcalQIECodersGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

HcalQIECodersGPU::HcalQIECodersGPU(HcalQIEData const& qiedata)
    : totalChannels_{qiedata.getAllContainers()[0].second.size() + qiedata.getAllContainers()[1].second.size()},
      offsets_(totalChannels_ * numValuesPerChannel),
      slopes_(totalChannels_ * numValuesPerChannel) {
  auto const containers = qiedata.getAllContainers();

  // fill in hb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    for (uint32_t k = 0; k < 4; k++)
      for (uint32_t l = 0; l < 4; l++) {
        auto const linear = k * 4 + l;
        offsets_[i * numValuesPerChannel + linear] = barrelValues[i].offset(k, l);
        slopes_[i * numValuesPerChannel + linear] = barrelValues[i].slope(k, l);
      }
  }

  // fill in he
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    auto const off = (i + offset) * numValuesPerChannel;
    for (uint32_t k = 0; k < 4; k++)
      for (uint32_t l = 0; l < 4; l++) {
        auto const linear = k * 4u + l;
        offsets_[off + linear] = endcapValues[i].offset(k, l);
        slopes_[off + linear] = endcapValues[i].slope(k, l);
      }
  }
}

HcalQIECodersGPU::Product const& HcalQIECodersGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalQIECodersGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.offsets = cms::cuda::make_device_unique<float[]>(offsets_.size(), stream);
        product.slopes = cms::cuda::make_device_unique<float[]>(slopes_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.offsets, offsets_, stream);
        cms::cuda::copyAsync(product.slopes, slopes_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalQIECodersGPU);
