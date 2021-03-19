#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParametersGPU.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

HcalSiPMParametersGPU::HcalSiPMParametersGPU(HcalSiPMParameters const& parameters)
    : totalChannels_{parameters.getAllContainers()[0].second.size() + parameters.getAllContainers()[1].second.size()},
      type_(totalChannels_),
      auxi1_(totalChannels_),
      fcByPE_(totalChannels_),
      darkCurrent_(totalChannels_),
      auxi2_(totalChannels_) {
  auto const containers = parameters.getAllContainers();

  // fill in eb
  auto const& barrelValues = containers[0].second;
  for (uint64_t i = 0; i < barrelValues.size(); ++i) {
    auto const& item = barrelValues[i];
    type_[i] = item.getType();
    auxi1_[i] = item.getauxi1();
    fcByPE_[i] = item.getFCByPE();
    darkCurrent_[i] = item.getDarkCurrent();
    auxi2_[i] = item.getauxi2();
  }

  // fill in ee
  auto const& endcapValues = containers[1].second;
  auto const offset = barrelValues.size();
  for (uint64_t i = 0; i < endcapValues.size(); ++i) {
    auto const off = offset + i;
    auto const& item = endcapValues[i];
    type_[off] = item.getType();
    auxi1_[off] = item.getauxi1();
    fcByPE_[off] = item.getFCByPE();
    darkCurrent_[off] = item.getDarkCurrent();
    auxi2_[off] = item.getauxi2();
  }
}

HcalSiPMParametersGPU::Product const& HcalSiPMParametersGPU::getProduct(cudaStream_t stream) const {
  auto const& product =
      product_.dataForCurrentDeviceAsync(stream, [this](HcalSiPMParametersGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.type = cms::cuda::make_device_unique<int[]>(type_.size(), stream);
        product.auxi1 = cms::cuda::make_device_unique<int[]>(auxi1_.size(), stream);
        product.fcByPE = cms::cuda::make_device_unique<float[]>(fcByPE_.size(), stream);
        product.darkCurrent = cms::cuda::make_device_unique<float[]>(darkCurrent_.size(), stream);
        product.auxi2 = cms::cuda::make_device_unique<float[]>(auxi2_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.type, type_, stream);
        cms::cuda::copyAsync(product.auxi1, auxi1_, stream);
        cms::cuda::copyAsync(product.fcByPE, fcByPE_, stream);
        cms::cuda::copyAsync(product.darkCurrent, darkCurrent_, stream);
        cms::cuda::copyAsync(product.auxi2, auxi2_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalSiPMParametersGPU);
