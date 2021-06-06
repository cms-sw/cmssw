#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsGPU.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

HcalSiPMCharacteristicsGPU::HcalSiPMCharacteristicsGPU(HcalSiPMCharacteristics const& parameters)
    : pixels_(parameters.getTypes()),
      auxi1_(parameters.getTypes()),
      parLin1_(parameters.getTypes()),
      parLin2_(parameters.getTypes()),
      parLin3_(parameters.getTypes()),
      crossTalk_(parameters.getTypes()),
      auxi2_(parameters.getTypes()) {
  for (uint32_t i = 0; i < parameters.getTypes(); i++) {
    auto const type = parameters.getType(i);
#ifdef HCAL_MAHI_CPUDEBUG
    printf("index = %u type = %d\n", i, type);
#endif

    // for now...
    if (static_cast<uint32_t>(type) != i + 1)
      throw cms::Exception("HcalSiPMCharacteristics")
          << "Wrong assumption for HcalSiPMcharacteristics type values, "
          << "should be type value <- type index + 1" << std::endl
          << "Observed type value = " << type << " and index = " << i << std::endl;

    pixels_[i] = parameters.getPixels(type);
    auxi1_[i] = parameters.getAuxi1(type);
    parLin1_[i] = parameters.getNonLinearities(type)[0];
    parLin2_[i] = parameters.getNonLinearities(type)[1];
    parLin3_[i] = parameters.getNonLinearities(type)[2];
    crossTalk_[i] = parameters.getCrossTalk(type);
    auxi2_[i] = parameters.getAuxi2(type);
  }
}

HcalSiPMCharacteristicsGPU::Product const& HcalSiPMCharacteristicsGPU::getProduct(cudaStream_t stream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      stream, [this](HcalSiPMCharacteristicsGPU::Product& product, cudaStream_t stream) {
        // allocate
        product.pixels = cms::cuda::make_device_unique<int[]>(pixels_.size(), stream);
        product.auxi1 = cms::cuda::make_device_unique<int[]>(auxi1_.size(), stream);
        product.parLin1 = cms::cuda::make_device_unique<float[]>(parLin1_.size(), stream);
        product.parLin2 = cms::cuda::make_device_unique<float[]>(parLin2_.size(), stream);
        product.parLin3 = cms::cuda::make_device_unique<float[]>(parLin3_.size(), stream);
        product.crossTalk = cms::cuda::make_device_unique<float[]>(crossTalk_.size(), stream);
        product.auxi2 = cms::cuda::make_device_unique<float[]>(auxi2_.size(), stream);

        // transfer
        cms::cuda::copyAsync(product.pixels, pixels_, stream);
        cms::cuda::copyAsync(product.auxi1, auxi1_, stream);
        cms::cuda::copyAsync(product.parLin1, parLin1_, stream);
        cms::cuda::copyAsync(product.parLin2, parLin2_, stream);
        cms::cuda::copyAsync(product.parLin3, parLin3_, stream);
        cms::cuda::copyAsync(product.crossTalk, crossTalk_, stream);
        cms::cuda::copyAsync(product.auxi2, auxi2_, stream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalSiPMCharacteristicsGPU);
