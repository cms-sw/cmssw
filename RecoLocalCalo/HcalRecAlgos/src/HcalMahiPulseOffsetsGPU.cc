#include "RecoLocalCalo/HcalRecAlgos/interface/HcalMahiPulseOffsetsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalMahiPulseOffsetsGPU::HcalMahiPulseOffsetsGPU(std::vector<int> const& values) {
  values_.resize(values.size());
  std::copy(values.begin(), values.end(), values_.begin());
}

HcalMahiPulseOffsetsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

HcalMahiPulseOffsetsGPU::Product const& HcalMahiPulseOffsetsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HcalMahiPulseOffsetsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->values_.data(),
                                  this->values_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HcalMahiPulseOffsetsGPU);
