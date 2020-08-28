#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMParametersGPU.h"

#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

HcalSiPMParametersGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(type));
  cudaCheck(cudaFree(auxi1));
  cudaCheck(cudaFree(fcByPE));
  cudaCheck(cudaFree(darkCurrent));
  cudaCheck(cudaFree(auxi2));
}

HcalSiPMParametersGPU::Product const& HcalSiPMParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalSiPMParametersGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.type, this->type_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.auxi1, this->auxi1_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.fcByPE, this->fcByPE_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.darkCurrent, this->darkCurrent_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.auxi2, this->auxi2_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(
            product.type, this->type_.data(), this->type_.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(
            product.auxi1, this->auxi1_.data(), this->auxi1_.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(product.fcByPE,
                                  this->fcByPE_.data(),
                                  this->fcByPE_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.darkCurrent,
                                  this->darkCurrent_.data(),
                                  this->darkCurrent_.size() * sizeof(float),
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

TYPELOOKUP_DATA_REG(HcalSiPMParametersGPU);
