#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalGainRatiosGPU::EcalGainRatiosGPU(EcalGainRatios const& values)
    : gain12Over6_(values.size()), gain6Over1_(values.size()) {
  // fill in eb
  auto const& barrelValues = values.barrelItems();
  for (unsigned int i = 0; i < barrelValues.size(); i++) {
    gain12Over6_[i] = barrelValues[i].gain12Over6();
    gain6Over1_[i] = barrelValues[i].gain6Over1();
  }

  // fill in ee
  auto const& endcapValues = values.endcapItems();
  auto const offset = barrelValues.size();
  for (unsigned int i = 0; i < endcapValues.size(); i++) {
    gain12Over6_[offset + i] = endcapValues[i].gain12Over6();
    gain6Over1_[offset + i] = endcapValues[i].gain6Over1();
  }
}

EcalGainRatiosGPU::Product const& EcalGainRatiosGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalGainRatiosGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.gain12Over6 = cms::cuda::make_device_unique<float[]>(gain12Over6_.size(), cudaStream);
        product.gain6Over1 = cms::cuda::make_device_unique<float[]>(gain6Over1_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.gain12Over6, gain12Over6_, cudaStream);
        cms::cuda::copyAsync(product.gain6Over1, gain6Over1_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalGainRatiosGPU);
