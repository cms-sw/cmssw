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
  auto tkidGPU = cms::cuda::make_device_unique<caConstants::tindex_type[]>(maxNumberOfConcurrentFits_, stream);
  auto hitsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double), stream);
  auto hits_geGPU = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float), stream);
  auto fast_fit_resultsGPU = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernel_BLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tuples_,
                                                                  tupleMultiplicity_,
                                                                  hv,
                                                                  tkidGPU.get(),
                                                                  hitsGPU.get(),
                                                                  hits_geGPU.get(),
                                                                  fast_fit_resultsGPU.get(),
                                                                  3,
                                                                  3,
                                                                  offset);
    cudaCheck(cudaGetLastError());

    kernel_BLFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                              bField_,
                                                              outputSoa_,
                                                              tkidGPU.get(),
                                                              hitsGPU.get(),
                                                              hits_geGPU.get(),
                                                              fast_fit_resultsGPU.get());
    cudaCheck(cudaGetLastError());

    if (fitNas4_) {
      // fit all as 4
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        8,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());
    } else {
      // fit quads
      kernel_BLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        4,
                                                                        4,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                                    bField_,
                                                                    outputSoa_,
                                                                    tkidGPU.get(),
                                                                    hitsGPU.get(),
                                                                    hits_geGPU.get(),
                                                                    fast_fit_resultsGPU.get());
      // fit penta (all 5)
      kernel_BLFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tuples_,
                                                                        tupleMultiplicity_,
                                                                        hv,
                                                                        tkidGPU.get(),
                                                                        hitsGPU.get(),
                                                                        hits_geGPU.get(),
                                                                        fast_fit_resultsGPU.get(),
                                                                        5,
                                                                        5,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<5><<<8, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
      // fit sexta and above (as 6)
      kernel_BLFastFit<6><<<4, blockSize, 0, stream>>>(tuples_,
                                                       tupleMultiplicity_,
                                                       hv,
                                                       tkidGPU.get(),
                                                       hitsGPU.get(),
                                                       hits_geGPU.get(),
                                                       fast_fit_resultsGPU.get(),
                                                       6,
                                                       8,
                                                       offset);
      cudaCheck(cudaGetLastError());

      kernel_BLFit<6><<<4, blockSize, 0, stream>>>(tupleMultiplicity_,
                                                   bField_,
                                                   outputSoa_,
                                                   tkidGPU.get(),
                                                   hitsGPU.get(),
                                                   hits_geGPU.get(),
                                                   fast_fit_resultsGPU.get());
      cudaCheck(cudaGetLastError());
    }

  }  // loop on concurrent fits
}
