#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

PFHBHERecHitParamsGPU::PFHBHERecHitParamsGPU(edm::ParameterSet const& ps) {
  auto const& valuesDepthHB = ps.getParameter<std::vector<int>>("depthHB");
  auto const& valuesDepthHE = ps.getParameter<std::vector<int>>("depthHE");
  auto const& valuesThresholdE_HB = ps.getParameter<std::vector<double>>("thresholdE_HB");
  auto const& valuesThresholdE_HE = ps.getParameter<std::vector<double>>("thresholdE_HE");
  depthHB_.resize(valuesDepthHB.size());
  depthHE_.resize(valuesDepthHE.size());
  thresholdE_HB_.resize(valuesThresholdE_HB.size());
  thresholdE_HE_.resize(valuesThresholdE_HE.size());
  std::copy(valuesDepthHB.begin(), valuesDepthHB.end(), depthHB_.begin());
  std::copy(valuesDepthHE.begin(), valuesDepthHE.end(), depthHE_.begin());
  std::copy(valuesThresholdE_HB.begin(), valuesThresholdE_HB.end(), thresholdE_HB_.begin());
  std::copy(valuesThresholdE_HE.begin(), valuesThresholdE_HE.end(), thresholdE_HE_.begin());
  nDepthHB_ = (int)valuesDepthHB.size();
  nDepthHE_ = (int)valuesDepthHE.size();
  if (valuesDepthHB.size() != valuesThresholdE_HB.size() || valuesDepthHE.size() != valuesThresholdE_HE.size())
    throw cms::Exception("PFHBHERecHitParamsGPU ")
        << "The parameter vector sizes are not consistent.\n"
        << " HB: " << valuesDepthHB.size() << " " << valuesThresholdE_HB.size() << " HE: " << valuesDepthHE.size()
        << " " << valuesThresholdE_HE.size() << "\n";
}

PFHBHERecHitParamsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(nDepthHB));
  cudaCheck(cudaFree(nDepthHE));
  cudaCheck(cudaFree(depthHB));
  cudaCheck(cudaFree(depthHE));
  cudaCheck(cudaFree(thresholdE_HB));
  cudaCheck(cudaFree(thresholdE_HE));
}

PFHBHERecHitParamsGPU::Product const& PFHBHERecHitParamsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](PFHBHERecHitParamsGPU::Product& product, cudaStream_t cudaStream) {
        //
        product.nDepthHB = cms::cuda::make_device_unique<int>(cudaStream);
        product.nDepthHE = cms::cuda::make_device_unique<int>(cudaStream);
        product.depthHB = cms::cuda::make_device_unique<int[]>(depthHB_.size(), cudaStream);
        product.depthHE = cms::cuda::make_device_unique<int[]>(depthHE_.size(), cudaStream);
        product.thresholdE_HB = cms::cuda::make_device_unique<float[]>(thresholdE_HB_.size(), cudaStream);
        product.thresholdE_HE = cms::cuda::make_device_unique<float[]>(thresholdE_HE_.size(), cudaStream);
        //
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.valuesdepthHB, this->valuesdepthHB_.size() * sizeof(int)));
        //cudaCheck(cudaMalloc((void**)&product.depthHE, this->depthHE_.size() * sizeof(int)));
        //cudaCheck(cudaMalloc((void**)&product.thresholdE_HB, this->thresholdE_HB_.size() * sizeof(float)));
        //cudaCheck(cudaMalloc((void**)&product.thresholdE_HE, this->thresholdE_HE_.size() * sizeof(float)));

        // transfer
        //add this copy syntax to https://cmssdt.cern.ch/lxr/source/HeterogeneousCore/CUDAUtilities/interface/copyAsync.h
        //so that we can use copyAsync for all transfers?
        cms::cuda::host::unique_ptr<int> nDepthHB_h = cms::cuda::make_host_unique<int>(cudaStream);
        cms::cuda::host::unique_ptr<int> nDepthHE_h = cms::cuda::make_host_unique<int>(cudaStream);
        *nDepthHB_h = nDepthHB_;
        *nDepthHE_h = nDepthHE_;
        cudaCheck(cudaMemcpyAsync(product.nDepthHB, nDepthHB_h.get(), sizeof(int), cudaMemcpyHostToDevice, cudaStream));
        cudaCheck(cudaMemcpyAsync(product.nDepthHE, nDepthHE_h.get(), sizeof(int), cudaMemcpyHostToDevice, cudaStream));
        //
        cms::cuda::copyAsync(product.depthHB, depthHB_, cudaStream);
        cms::cuda::copyAsync(product.depthHE, depthHE_, cudaStream);
        cms::cuda::copyAsync(product.thresholdE_HB, thresholdE_HB_, cudaStream);
        cms::cuda::copyAsync(product.thresholdE_HE, thresholdE_HE_, cudaStream);
        //
        /*
        cudaCheck(cudaMemcpyAsync(product.valuesdepthHB,
                                  this->valuesdepthHB_.data(),
                                  this->valuesdepthHB_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.depthHB,
                                  this->depthHE_.data(),
                                  this->depthHE_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.thresholdE_HB,
                                  this->thresholdE_HB_.data(),
                                  this->thresholdE_HB_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.thresholdE_HE,
                                  this->thresholdE_HE_.data(),
                                  this->thresholdE_HE_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
	*/
      });

  return product;
}

TYPELOOKUP_DATA_REG(PFHBHERecHitParamsGPU);
