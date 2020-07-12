#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMCharacteristicsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "FWCore/Utilities/interface/Exception.h"

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

HcalSiPMCharacteristicsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(pixels));
  cudaCheck(cudaFree(auxi1));
  cudaCheck(cudaFree(parLin1));
  cudaCheck(cudaFree(parLin2));
  cudaCheck(cudaFree(parLin3));
  cudaCheck(cudaFree(crossTalk));
  cudaCheck(cudaFree(auxi2));
}

HcalSiPMCharacteristicsGPU::Product const& HcalSiPMCharacteristicsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalSiPMCharacteristicsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.pixels, this->pixels_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.auxi1, this->auxi1_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.parLin1, this->parLin1_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.parLin2, this->parLin2_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.parLin3, this->parLin3_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.crossTalk, this->crossTalk_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.auxi2, this->auxi2_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.pixels,
                                  this->pixels_.data(),
                                  this->pixels_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(
            product.auxi1, this->auxi1_.data(), this->auxi1_.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(product.parLin1,
                                  this->parLin1_.data(),
                                  this->parLin1_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.parLin2,
                                  this->parLin2_.data(),
                                  this->parLin2_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.parLin3,
                                  this->parLin3_.data(),
                                  this->parLin3_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.crossTalk,
                                  this->crossTalk_.data(),
                                  this->crossTalk_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.auxi2,
                                  this->auxi2_.data(),
                                  this->auxi2_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalSiPMCharacteristicsGPU);
