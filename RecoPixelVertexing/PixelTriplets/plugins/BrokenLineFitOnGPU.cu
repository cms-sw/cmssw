#include "BrokenLineFitOnGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream) {
  assert(tuples_);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernel_BLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(
        tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);
    cudaCheck(cudaGetLastError());

    kernel_BLFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                              bField_,
                                                              outputSoa_,
                                                              hitsGPU_.get(),
                                                              hits_geGPU_.get(),
                                                              fast_fit_resultsGPU_.get(),
                                                              3,
                                                              offset);
    cudaCheck(cudaGetLastError());

    // fit quads
    kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
        tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);
    cudaCheck(cudaGetLastError());

    kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                  bField_,
                                                                  outputSoa_,
                                                                  hitsGPU_.get(),
                                                                  hits_geGPU_.get(),
                                                                  fast_fit_resultsGPU_.get(),
                                                                  4,
                                                                  offset);
    cudaCheck(cudaGetLastError());

    if (fit5as4_) {
      // fit penta (only first 4)
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    5,
                                                                    offset);
      cudaCheck(cudaGetLastError());
    } else {
      // fit penta (all 5)
      kernel_BLFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    5,
                                                                    offset);
      cudaCheck(cudaGetLastError());
    }

  }  // loop on concurrent fits
}
