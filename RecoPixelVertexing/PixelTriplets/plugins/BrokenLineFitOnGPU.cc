#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernelsOnCPU(HitsView const* hv, uint32_t hitsInFit, uint32_t maxNumberOfTuples) {
  assert(tuples_d);

  //  Fit internals
  auto hitsGPU_ = std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU_ = std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU_ =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);

    kernelBLFit<3>(tupleMultiplicity_d,
                   bField_,
                   outputSoa_d,
                   hitsGPU_.get(),
                   hits_geGPU_.get(),
                   fast_fit_resultsGPU_.get(),
                   3,
                   offset);

    // fit quads
    kernelBLFastFit<4>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);

    kernelBLFit<4>(tupleMultiplicity_d,
                   bField_,
                   outputSoa_d,
                   hitsGPU_.get(),
                   hits_geGPU_.get(),
                   fast_fit_resultsGPU_.get(),
                   4,
                   offset);

    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);

      kernelBLFit<4>(tupleMultiplicity_d,
                     bField_,
                     outputSoa_d,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     5,
                     offset);
    } else {
      // fit penta (all 5)
      kernelBLFastFit<5>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);

      kernelBLFit<5>(tupleMultiplicity_d,
                     bField_,
                     outputSoa_d,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     5,
                     offset);
    }

  }  // loop on concurrent fits
}
