#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

// FIXME: add proper getters to conditions
HcalRecoParamsGPU::HcalRecoParamsGPU(HcalRecoParams const& recoParams)
    : totalChannels_{recoParams.getAllContainers()[0].second.size() + recoParams.getAllContainers()[1].second.size()},
      param1_(totalChannels_),
      param2_(totalChannels_) {
  auto const& containers = recoParams.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    param1_[i] = barrelValues[i].param1();
    param2_[i] = barrelValues[i].param2();
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    param1_[i + offset] = endcapValues[i].param1();
    param2_[i + offset] = endcapValues[i].param2();
  }
}

HcalRecoParamsGPU::Product const& HcalRecoParamsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalRecoParamsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.param1 = cms::cuda::make_device_unique<uint32_t[]>(param1_.size(), stream);
        product.param2 = cms::cuda::make_device_unique<uint32_t[]>(param2_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.param1, param1_, stream);
        cms::cuda::copyAsync(product.param2, param2_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalRecoParamsGPU);
