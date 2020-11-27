#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPedestalsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

EcalPedestalsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(mean_x12));
  cudaCheck(cudaFree(rms_x12));
  cudaCheck(cudaFree(mean_x6));
  cudaCheck(cudaFree(rms_x6));
  cudaCheck(cudaFree(mean_x1));
  cudaCheck(cudaFree(rms_x1));
}

EcalPedestalsGPU::Product const& EcalPedestalsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalPedestalsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.mean_x12, this->mean_x12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.rms_x12, this->mean_x12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.mean_x6, this->mean_x12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.rms_x6, this->mean_x12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.mean_x1, this->mean_x12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.rms_x1, this->mean_x12_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.mean_x12,
                                  this->mean_x12_.data(),
                                  this->mean_x12_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.rms_x12,
                                  this->rms_x12_.data(),
                                  this->rms_x12_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.mean_x6,
                                  this->mean_x6_.data(),
                                  this->mean_x6_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.rms_x6,
                                  this->rms_x6_.data(),
                                  this->rms_x6_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.mean_x1,
                                  this->mean_x1_.data(),
                                  this->mean_x1_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.rms_x1,
                                  this->rms_x1_.data(),
                                  this->rms_x1_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalPedestalsGPU);
