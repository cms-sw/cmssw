#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

PFHBHERecHitParamsGPU::PFHBHERecHitParamsGPU(edm::ParameterSet const& ps) {
  auto const& valuesdepthHB = ps.getParameter<std::vector<int>>("depthHB");
  auto const& valuesdepthHE = ps.getParameter<std::vector<int>>("depthHE");
  auto const& valuesthresholdE_HB = ps.getParameter<std::vector<double>>("thresholdE_HB");
  auto const& valuesthresholdE_HE = ps.getParameter<std::vector<double>>("thresholdE_HE");
  valuesdepthHB_.resize(valuesdepthHB.size());
  valuesdepthHE_.resize(valuesdepthHE.size());
  valuesthresholdE_HB_.resize(valuesthresholdE_HB.size());
  valuesthresholdE_HE_.resize(valuesthresholdE_HE.size());
  std::copy(valuesdepthHB.begin(), valuesdepthHB.end(), valuesdepthHB_.begin());
  std::copy(valuesdepthHE.begin(), valuesdepthHE.end(), valuesdepthHE_.begin());
  std::copy(valuesthresholdE_HB.begin(), valuesthresholdE_HB.end(), valuesthresholdE_HB_.begin());
  std::copy(valuesthresholdE_HE.begin(), valuesthresholdE_HE.end(), valuesthresholdE_HE_.begin());
}

PFHBHERecHitParamsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(valuesdepthHB));
  cudaCheck(cudaFree(valuesdepthHE));
  cudaCheck(cudaFree(valuesthresholdE_HB));
  cudaCheck(cudaFree(valuesthresholdE_HE));
}

PFHBHERecHitParamsGPU::Product const& PFHBHERecHitParamsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](PFHBHERecHitParamsGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.valuesdepthHB, this->valuesdepthHB_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.valuesdepthHE, this->valuesdepthHE_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.valuesthresholdE_HB, this->valuesthresholdE_HB_.size() * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&product.valuesthresholdE_HE, this->valuesthresholdE_HE_.size() * sizeof(float)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.valuesdepthHB,
                                  this->valuesdepthHB_.data(),
                                  this->valuesdepthHB_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.valuesdepthHE,
                                  this->valuesdepthHE_.data(),
                                  this->valuesdepthHE_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.valuesthresholdE_HB,
                                  this->valuesthresholdE_HB_.data(),
                                  this->valuesthresholdE_HB_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.valuesthresholdE_HE,
                                  this->valuesthresholdE_HE_.data(),
                                  this->valuesthresholdE_HE_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(PFHBHERecHitParamsGPU);
