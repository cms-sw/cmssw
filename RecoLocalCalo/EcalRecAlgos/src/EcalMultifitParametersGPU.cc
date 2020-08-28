#include "RecoLocalCalo/EcalRecAlgos/interface/EcalMultifitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalMultifitParametersGPU::EcalMultifitParametersGPU(edm::ParameterSet const& ps) {
  auto const& amplitudeFitParametersEB = ps.getParameter<std::vector<double>>("EBamplitudeFitParameters");
  auto const& amplitudeFitParametersEE = ps.getParameter<std::vector<double>>("EEamplitudeFitParameters");
  auto const& timeFitParametersEB = ps.getParameter<std::vector<double>>("EBtimeFitParameters");
  auto const& timeFitParametersEE = ps.getParameter<std::vector<double>>("EEtimeFitParameters");

  amplitudeFitParametersEB_.resize(amplitudeFitParametersEB.size());
  amplitudeFitParametersEE_.resize(amplitudeFitParametersEE.size());
  timeFitParametersEB_.resize(timeFitParametersEB.size());
  timeFitParametersEE_.resize(timeFitParametersEE.size());

  std::copy(amplitudeFitParametersEB.begin(), amplitudeFitParametersEB.end(), amplitudeFitParametersEB_.begin());
  std::copy(amplitudeFitParametersEE.begin(), amplitudeFitParametersEE.end(), amplitudeFitParametersEE_.begin());
  std::copy(timeFitParametersEB.begin(), timeFitParametersEB.end(), timeFitParametersEB_.begin());
  std::copy(timeFitParametersEE.begin(), timeFitParametersEE.end(), timeFitParametersEE_.begin());
}

EcalMultifitParametersGPU::Product::~Product() {
  cudaCheck(cudaFree(amplitudeFitParametersEB));
  cudaCheck(cudaFree(amplitudeFitParametersEE));
  cudaCheck(cudaFree(timeFitParametersEB));
  cudaCheck(cudaFree(timeFitParametersEE));
}

EcalMultifitParametersGPU::Product const& EcalMultifitParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalMultifitParametersGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.amplitudeFitParametersEB,
                             this->amplitudeFitParametersEB_.size() * sizeof(double)));
        cudaCheck(cudaMalloc((void**)&product.amplitudeFitParametersEE,
                             this->amplitudeFitParametersEE_.size() * sizeof(double)));
        cudaCheck(cudaMalloc((void**)&product.timeFitParametersEB, this->timeFitParametersEB_.size() * sizeof(double)));
        cudaCheck(cudaMalloc((void**)&product.timeFitParametersEE, this->timeFitParametersEE_.size() * sizeof(double)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.amplitudeFitParametersEB,
                                  this->amplitudeFitParametersEB_.data(),
                                  this->amplitudeFitParametersEB_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.amplitudeFitParametersEE,
                                  this->amplitudeFitParametersEE_.data(),
                                  this->amplitudeFitParametersEE_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.timeFitParametersEB,
                                  this->timeFitParametersEB_.data(),
                                  this->timeFitParametersEB_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.timeFitParametersEE,
                                  this->timeFitParametersEE_.data(),
                                  this->timeFitParametersEE_.size() * sizeof(double),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });
  return product;
}

TYPELOOKUP_DATA_REG(EcalMultifitParametersGPU);
