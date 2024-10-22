#include "CondFormats/EcalObjects/interface/EcalPedestalsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalPedestalsGPU::EcalPedestalsGPU(EcalPedestals const& pedestals)
    : mean_x12_(pedestals.size()),
      rms_x12_(pedestals.size()),
      mean_x6_(pedestals.size()),
      rms_x6_(pedestals.size()),
      mean_x1_(pedestals.size()),
      rms_x1_(pedestals.size()) {
  // fill in eb
  auto const& barrelValues = pedestals.barrelItems();
  for (unsigned int i = 0; i < barrelValues.size(); i++) {
    mean_x12_[i] = barrelValues[i].mean_x12;
    rms_x12_[i] = barrelValues[i].rms_x12;
    mean_x6_[i] = barrelValues[i].mean_x6;
    rms_x6_[i] = barrelValues[i].rms_x6;
    mean_x1_[i] = barrelValues[i].mean_x1;
    rms_x1_[i] = barrelValues[i].rms_x1;
  }

  // fill in ee
  auto const& endcapValues = pedestals.endcapItems();
  auto const offset = barrelValues.size();
  for (unsigned int i = 0; i < endcapValues.size(); i++) {
    mean_x12_[offset + i] = endcapValues[i].mean_x12;
    rms_x12_[offset + i] = endcapValues[i].rms_x12;
    mean_x6_[offset + i] = endcapValues[i].mean_x6;
    rms_x6_[offset + i] = endcapValues[i].rms_x6;
    mean_x1_[offset + i] = endcapValues[i].mean_x1;
    rms_x1_[offset + i] = endcapValues[i].rms_x1;
  }
}

EcalPedestalsGPU::Product const& EcalPedestalsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalPedestalsGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.mean_x12 = cms::cuda::make_device_unique<float[]>(mean_x12_.size(), cudaStream);
        product.mean_x6 = cms::cuda::make_device_unique<float[]>(mean_x6_.size(), cudaStream);
        product.mean_x1 = cms::cuda::make_device_unique<float[]>(mean_x1_.size(), cudaStream);
        product.rms_x12 = cms::cuda::make_device_unique<float[]>(rms_x12_.size(), cudaStream);
        product.rms_x6 = cms::cuda::make_device_unique<float[]>(rms_x6_.size(), cudaStream);
        product.rms_x1 = cms::cuda::make_device_unique<float[]>(rms_x1_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.mean_x12, mean_x12_, cudaStream);
        cms::cuda::copyAsync(product.mean_x6, mean_x6_, cudaStream);
        cms::cuda::copyAsync(product.mean_x1, mean_x1_, cudaStream);
        cms::cuda::copyAsync(product.rms_x12, rms_x12_, cudaStream);
        cms::cuda::copyAsync(product.rms_x6, rms_x6_, cudaStream);
        cms::cuda::copyAsync(product.rms_x1, rms_x1_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalPedestalsGPU);
