#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalWidthsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

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

HcalPedestalWidthsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(sigma00));
  cudaCheck(cudaFree(sigma01));
  cudaCheck(cudaFree(sigma02));
  cudaCheck(cudaFree(sigma03));
  cudaCheck(cudaFree(sigma10));
  cudaCheck(cudaFree(sigma11));
  cudaCheck(cudaFree(sigma12));
  cudaCheck(cudaFree(sigma13));
  cudaCheck(cudaFree(sigma20));
  cudaCheck(cudaFree(sigma21));
  cudaCheck(cudaFree(sigma22));
  cudaCheck(cudaFree(sigma23));
  cudaCheck(cudaFree(sigma30));
  cudaCheck(cudaFree(sigma31));
  cudaCheck(cudaFree(sigma32));
  cudaCheck(cudaFree(sigma33));
}

HcalPedestalWidthsGPU::Product const& HcalPedestalWidthsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalPedestalWidthsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.sigma00, this->sigma00_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma01, this->sigma01_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma02, this->sigma02_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma03, this->sigma03_.size() * sizeof(float)));

        cudaCheck(cudaMalloc((void**)&product.sigma10, this->sigma10_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma11, this->sigma11_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma12, this->sigma12_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma13, this->sigma13_.size() * sizeof(float)));

        cudaCheck(cudaMalloc((void**)&product.sigma20, this->sigma20_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma21, this->sigma21_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma22, this->sigma22_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma23, this->sigma23_.size() * sizeof(float)));

        cudaCheck(cudaMalloc((void**)&product.sigma30, this->sigma30_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma31, this->sigma31_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma32, this->sigma32_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.sigma33, this->sigma33_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.sigma00,
                                  this->sigma00_.data(),
                                  this->sigma00_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma01,
                                  this->sigma01_.data(),
                                  this->sigma01_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma02,
                                  this->sigma02_.data(),
                                  this->sigma02_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma03,
                                  this->sigma03_.data(),
                                  this->sigma03_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        cudaCheck(cudaMemcpyAsync(product.sigma10,
                                  this->sigma10_.data(),
                                  this->sigma10_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma11,
                                  this->sigma11_.data(),
                                  this->sigma11_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma12,
                                  this->sigma12_.data(),
                                  this->sigma12_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma13,
                                  this->sigma13_.data(),
                                  this->sigma13_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        cudaCheck(cudaMemcpyAsync(product.sigma20,
                                  this->sigma20_.data(),
                                  this->sigma20_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma21,
                                  this->sigma21_.data(),
                                  this->sigma21_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma22,
                                  this->sigma22_.data(),
                                  this->sigma22_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma23,
                                  this->sigma23_.data(),
                                  this->sigma23_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        cudaCheck(cudaMemcpyAsync(product.sigma30,
                                  this->sigma30_.data(),
                                  this->sigma30_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma31,
                                  this->sigma31_.data(),
                                  this->sigma31_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma32,
                                  this->sigma32_.data(),
                                  this->sigma32_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.sigma33,
                                  this->sigma33_.data(),
                                  this->sigma33_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalPedestalWidthsGPU);
