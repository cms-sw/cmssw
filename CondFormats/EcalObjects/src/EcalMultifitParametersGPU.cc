#include "CondFormats/EcalObjects/interface/EcalMultifitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalMultifitParametersGPU::EcalMultifitParametersGPU(std::vector<double> const& amplitudeFitParametersEB,
                                                     std::vector<double> const& amplitudeFitParametersEE,
                                                     std::vector<double> const& timeFitParametersEB,
                                                     std::vector<double> const& timeFitParametersEE) {
  amplitudeFitParametersEB_.resize(amplitudeFitParametersEB.size());
  amplitudeFitParametersEE_.resize(amplitudeFitParametersEE.size());
  timeFitParametersEB_.resize(timeFitParametersEB.size());
  timeFitParametersEE_.resize(timeFitParametersEE.size());

  std::copy(amplitudeFitParametersEB.begin(), amplitudeFitParametersEB.end(), amplitudeFitParametersEB_.begin());
  std::copy(amplitudeFitParametersEE.begin(), amplitudeFitParametersEE.end(), amplitudeFitParametersEE_.begin());
  std::copy(timeFitParametersEB.begin(), timeFitParametersEB.end(), timeFitParametersEB_.begin());
  std::copy(timeFitParametersEE.begin(), timeFitParametersEE.end(), timeFitParametersEE_.begin());
}

EcalMultifitParametersGPU::Product const& EcalMultifitParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalMultifitParametersGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.amplitudeFitParametersEB =
            cms::cuda::make_device_unique<double[]>(amplitudeFitParametersEB_.size(), cudaStream);
        product.amplitudeFitParametersEE =
            cms::cuda::make_device_unique<double[]>(amplitudeFitParametersEE_.size(), cudaStream);
        product.timeFitParametersEB = cms::cuda::make_device_unique<double[]>(timeFitParametersEB_.size(), cudaStream);
        product.timeFitParametersEE = cms::cuda::make_device_unique<double[]>(timeFitParametersEE_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.amplitudeFitParametersEB, amplitudeFitParametersEB_, cudaStream);
        cms::cuda::copyAsync(product.amplitudeFitParametersEE, amplitudeFitParametersEE_, cudaStream);
        cms::cuda::copyAsync(product.timeFitParametersEB, timeFitParametersEB_, cudaStream);
        cms::cuda::copyAsync(product.timeFitParametersEE, timeFitParametersEE_, cudaStream);
      });
  return product;
}

TYPELOOKUP_DATA_REG(EcalMultifitParametersGPU);
