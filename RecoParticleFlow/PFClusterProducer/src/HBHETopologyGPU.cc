#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

HBHETopologyGPU::HBHETopologyGPU(edm::ParameterSet const& ps) {
  //auto const& values = ps.getParameter<std::vector<int>>("pulseOffsets");
  //values_.resize(values.size());
  //std::copy(values.begin(), values.end(), values_.begin());
}

HBHETopologyGPU::Product::~Product() {
  // deallocation
  //cudaCheck(cudaFree(pos));
  cudaCheck(cudaFree(detId));
  cudaCheck(cudaFree(neighbours));
}

HBHETopologyGPU::Product const& HBHETopologyGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](HBHETopologyGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.values, this->values_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.detId, this->detId_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.neighbours, this->neighbours_.size() * sizeof(int)));

        // transfer
        // cudaCheck(cudaMemcpyAsync(product.values,
        //                           this->values_.data(),
        //                           this->values_.size() * sizeof(int),
        //                           cudaMemcpyHostToDevice,
        //                           cudaStream));
        cudaCheck(cudaMemcpyAsync(product.detId,
                                  this->detId_.data(),
                                  this->detId_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.neighbours,
                                  this->neighbours_.data(),
                                  this->neighbours_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(HBHETopologyGPU);
