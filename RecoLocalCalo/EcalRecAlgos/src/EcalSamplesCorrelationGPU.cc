#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSamplesCorrelationGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalSamplesCorrelationGPU::EcalSamplesCorrelationGPU(EcalSamplesCorrelation const& values)
    : EBG12SamplesCorrelation_{values.EBG12SamplesCorrelation},
      EBG6SamplesCorrelation_{values.EBG6SamplesCorrelation},
      EBG1SamplesCorrelation_{values.EBG1SamplesCorrelation},
      EEG12SamplesCorrelation_{values.EEG12SamplesCorrelation},
      EEG6SamplesCorrelation_{values.EEG6SamplesCorrelation},
      EEG1SamplesCorrelation_{values.EEG1SamplesCorrelation} {}

EcalSamplesCorrelationGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(EBG12SamplesCorrelation));
  cudaCheck(cudaFree(EBG6SamplesCorrelation));
  cudaCheck(cudaFree(EBG1SamplesCorrelation));
  cudaCheck(cudaFree(EEG12SamplesCorrelation));
  cudaCheck(cudaFree(EEG6SamplesCorrelation));
  cudaCheck(cudaFree(EEG1SamplesCorrelation));
}

EcalSamplesCorrelationGPU::Product const& EcalSamplesCorrelationGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalSamplesCorrelationGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.EBG12SamplesCorrelation,
                             this->EBG12SamplesCorrelation_.size() * sizeof(double)));
        cudaCheck(
            cudaMalloc((void**)&product.EBG6SamplesCorrelation, this->EBG6SamplesCorrelation_.size() * sizeof(double)));
        cudaCheck(
            cudaMalloc((void**)&product.EBG1SamplesCorrelation, this->EBG1SamplesCorrelation_.size() * sizeof(double)));
        cudaCheck(cudaMalloc((void**)&product.EEG12SamplesCorrelation,
                             this->EEG12SamplesCorrelation_.size() * sizeof(double)));
        cudaCheck(
            cudaMalloc((void**)&product.EEG6SamplesCorrelation, this->EEG6SamplesCorrelation_.size() * sizeof(double)));
        cudaCheck(
            cudaMalloc((void**)&product.EEG1SamplesCorrelation, this->EEG1SamplesCorrelation_.size() * sizeof(double)));
        // transfer
        cudaCheck(cudaMemcpyAsync(product.EBG12SamplesCorrelation,
                                  this->EBG12SamplesCorrelation_.data(),
                                  this->EBG12SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.EBG6SamplesCorrelation,
                                  this->EBG6SamplesCorrelation_.data(),
                                  this->EBG6SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.EBG1SamplesCorrelation,
                                  this->EBG1SamplesCorrelation_.data(),
                                  this->EBG1SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.EEG12SamplesCorrelation,
                                  this->EEG12SamplesCorrelation_.data(),
                                  this->EEG12SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.EEG6SamplesCorrelation,
                                  this->EEG6SamplesCorrelation_.data(),
                                  this->EEG6SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.EEG1SamplesCorrelation,
                                  this->EEG1SamplesCorrelation_.data(),
                                  this->EEG1SamplesCorrelation_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalSamplesCorrelationGPU);
