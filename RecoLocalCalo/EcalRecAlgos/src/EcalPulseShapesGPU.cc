#include "RecoLocalCalo/EcalRecAlgos/interface/EcalPulseShapesGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalPulseShapesGPU::EcalPulseShapesGPU(EcalPulseShapes const& values)
    : valuesEB_{values.barrelItems()}, valuesEE_{values.endcapItems()} {}

EcalPulseShapesGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(values));
}

EcalPulseShapesGPU::Product const& EcalPulseShapesGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalPulseShapesGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.values,
                             (this->valuesEE_.size() + this->valuesEB_.size()) * sizeof(EcalPulseShape)));

        // offset in terms of sizeof(EcalPulseShape) - plain c array
        uint32_t offset = this->valuesEB_.size();

        // transfer eb
        cudaCheck(cudaMemcpyAsync(product.values,
                                  this->valuesEB_.data(),
                                  this->valuesEB_.size() * sizeof(EcalPulseShape),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));

        // transfer ee starting at values + offset
        cudaCheck(cudaMemcpyAsync(product.values + offset,
                                  this->valuesEE_.data(),
                                  this->valuesEE_.size() * sizeof(EcalPulseShape),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalPulseShapesGPU);
