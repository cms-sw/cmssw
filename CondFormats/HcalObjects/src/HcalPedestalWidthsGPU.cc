#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidthsGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

// FIXME: add proper getters to conditions
HcalPedestalWidthsGPU::HcalPedestalWidthsGPU(HcalPedestalWidths const& pedestals)
    : unitIsADC_{pedestals.isADC()},
      totalChannels_{pedestals.getAllContainers()[0].second.size() + pedestals.getAllContainers()[1].second.size()},
      sigma00_(totalChannels_),
      sigma01_(totalChannels_),
      sigma02_(totalChannels_),
      sigma03_(totalChannels_),
      sigma10_(totalChannels_),
      sigma11_(totalChannels_),
      sigma12_(totalChannels_),
      sigma13_(totalChannels_),
      sigma20_(totalChannels_),
      sigma21_(totalChannels_),
      sigma22_(totalChannels_),
      sigma23_(totalChannels_),
      sigma30_(totalChannels_),
      sigma31_(totalChannels_),
      sigma32_(totalChannels_),
      sigma33_(totalChannels_) {
  auto const containers = pedestals.getAllContainers();

  // fill in hb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    sigma00_[i] = *(barrelValues[i].getValues() /* + 0 */);
    sigma01_[i] = *(barrelValues[i].getValues() + 1);
    sigma02_[i] = *(barrelValues[i].getValues() + 2);
    sigma03_[i] = *(barrelValues[i].getValues() + 3);
    sigma10_[i] = *(barrelValues[i].getValues() + 3);
    sigma11_[i] = *(barrelValues[i].getValues() + 5);
    sigma12_[i] = *(barrelValues[i].getValues() + 6);
    sigma13_[i] = *(barrelValues[i].getValues() + 7);
    sigma20_[i] = *(barrelValues[i].getValues() + 8);
    sigma21_[i] = *(barrelValues[i].getValues() + 9);
    sigma22_[i] = *(barrelValues[i].getValues() + 10);
    sigma23_[i] = *(barrelValues[i].getValues() + 11);
    sigma30_[i] = *(barrelValues[i].getValues() + 12);
    sigma31_[i] = *(barrelValues[i].getValues() + 13);
    sigma32_[i] = *(barrelValues[i].getValues() + 14);
    sigma33_[i] = *(barrelValues[i].getValues() + 15);
  }

  // fill in he
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    sigma00_[i + offset] = *(endcapValues[i].getValues() /* + 0 */);
    sigma01_[i + offset] = *(endcapValues[i].getValues() + 1);
    sigma02_[i + offset] = *(endcapValues[i].getValues() + 2);
    sigma03_[i + offset] = *(endcapValues[i].getValues() + 3);
    sigma10_[i + offset] = *(endcapValues[i].getValues() + 3);
    sigma11_[i + offset] = *(endcapValues[i].getValues() + 5);
    sigma12_[i + offset] = *(endcapValues[i].getValues() + 6);
    sigma13_[i + offset] = *(endcapValues[i].getValues() + 7);
    sigma20_[i + offset] = *(endcapValues[i].getValues() + 8);
    sigma21_[i + offset] = *(endcapValues[i].getValues() + 9);
    sigma22_[i + offset] = *(endcapValues[i].getValues() + 10);
    sigma23_[i + offset] = *(endcapValues[i].getValues() + 11);
    sigma30_[i + offset] = *(endcapValues[i].getValues() + 12);
    sigma31_[i + offset] = *(endcapValues[i].getValues() + 13);
    sigma32_[i + offset] = *(endcapValues[i].getValues() + 14);
    sigma33_[i + offset] = *(endcapValues[i].getValues() + 15);
  }
}

HcalPedestalWidthsGPU::Product const& HcalPedestalWidthsGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalPedestalWidthsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.sigma00 = cms::cuda::make_device_unique<float[]>(sigma00_.size(), stream);
        product.sigma01 = cms::cuda::make_device_unique<float[]>(sigma01_.size(), stream);
        product.sigma02 = cms::cuda::make_device_unique<float[]>(sigma02_.size(), stream);
        product.sigma03 = cms::cuda::make_device_unique<float[]>(sigma03_.size(), stream);

        product.sigma10 = cms::cuda::make_device_unique<float[]>(sigma10_.size(), stream);
        product.sigma11 = cms::cuda::make_device_unique<float[]>(sigma11_.size(), stream);
        product.sigma12 = cms::cuda::make_device_unique<float[]>(sigma12_.size(), stream);
        product.sigma13 = cms::cuda::make_device_unique<float[]>(sigma13_.size(), stream);

        product.sigma20 = cms::cuda::make_device_unique<float[]>(sigma20_.size(), stream);
        product.sigma21 = cms::cuda::make_device_unique<float[]>(sigma21_.size(), stream);
        product.sigma22 = cms::cuda::make_device_unique<float[]>(sigma22_.size(), stream);
        product.sigma23 = cms::cuda::make_device_unique<float[]>(sigma23_.size(), stream);

        product.sigma30 = cms::cuda::make_device_unique<float[]>(sigma30_.size(), stream);
        product.sigma31 = cms::cuda::make_device_unique<float[]>(sigma31_.size(), stream);
        product.sigma32 = cms::cuda::make_device_unique<float[]>(sigma32_.size(), stream);
        product.sigma33 = cms::cuda::make_device_unique<float[]>(sigma33_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.sigma00, sigma00_, stream);
        cms::cuda::copyAsync(product.sigma01, sigma01_, stream);
        cms::cuda::copyAsync(product.sigma02, sigma02_, stream);
        cms::cuda::copyAsync(product.sigma03, sigma03_, stream);

        cms::cuda::copyAsync(product.sigma10, sigma10_, stream);
        cms::cuda::copyAsync(product.sigma11, sigma11_, stream);
        cms::cuda::copyAsync(product.sigma12, sigma12_, stream);
        cms::cuda::copyAsync(product.sigma13, sigma13_, stream);

        cms::cuda::copyAsync(product.sigma20, sigma20_, stream);
        cms::cuda::copyAsync(product.sigma21, sigma21_, stream);
        cms::cuda::copyAsync(product.sigma22, sigma22_, stream);
        cms::cuda::copyAsync(product.sigma23, sigma23_, stream);

        cms::cuda::copyAsync(product.sigma30, sigma30_, stream);
        cms::cuda::copyAsync(product.sigma31, sigma31_, stream);
        cms::cuda::copyAsync(product.sigma32, sigma32_, stream);
        cms::cuda::copyAsync(product.sigma33, sigma33_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalPedestalWidthsGPU);
