#include "RecoLocalCalo/EcalRecAlgos/interface/EcalGainRatiosGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

EcalGainRatiosGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(gain12Over6));
  cudaCheck(cudaFree(gain6Over1));
}

EcalGainRatiosGPU::Product const& EcalGainRatiosGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalGainRatiosGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.gain12Over6, this->gain12Over6_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.gain6Over1, this->gain6Over1_.size() * sizeof(float)));
        // transfer
        cudaCheck(cudaMemcpyAsync(product.gain12Over6,
                                  this->gain12Over6_.data(),
                                  this->gain12Over6_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.gain6Over1,
                                  this->gain6Over1_.data(),
                                  this->gain6Over1_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalGainRatiosGPU);
