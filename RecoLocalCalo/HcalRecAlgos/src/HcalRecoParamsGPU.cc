#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

HcalRecoParamsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(param1));
  cudaCheck(cudaFree(param2));
}

HcalRecoParamsGPU::Product const& HcalRecoParamsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalRecoParamsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.param1, this->param1_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.param2, this->param2_.size() * sizeof(uint32_t)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.param1,
                                  this->param1_.data(),
                                  this->param1_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.param2,
                                  this->param2_.data(),
                                  this->param2_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalRecoParamsGPU);
