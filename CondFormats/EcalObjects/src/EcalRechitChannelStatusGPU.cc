#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalRechitChannelStatusGPU::EcalRechitChannelStatusGPU(EcalChannelStatus const& values) : status_(values.size()) {
  // fill in eb
  auto const& barrelValues = values.barrelItems();
  for (unsigned int i = 0; i < barrelValues.size(); i++) {
    status_[i] = barrelValues[i].getEncodedStatusCode();
  }

  // fill in ee
  auto const& endcapValues = values.endcapItems();
  auto const offset = barrelValues.size();
  for (unsigned int i = 0; i < endcapValues.size(); i++) {
    status_[offset + i] = endcapValues[i].getEncodedStatusCode();
  }
}

EcalRechitChannelStatusGPU::Product const& EcalRechitChannelStatusGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalRechitChannelStatusGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.status = cms::cuda::make_device_unique<uint16_t[]>(status_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.status, status_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalRechitChannelStatusGPU);
