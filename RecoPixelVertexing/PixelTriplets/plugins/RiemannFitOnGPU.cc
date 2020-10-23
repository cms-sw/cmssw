#include "RiemannFitOnGPU.h"

void HelixFitOnGPU::launchRiemannKernelsOnCPU(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples) {
  assert(tuples_d);

  //  Fit internals
  auto hitsGPU_ = std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU_ = std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU_ =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));
  auto circle_fit_resultsGPU_holder = std::make_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit));
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3>(
        tuples_d, tupleMultiplicity_d, 3, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);

    kernelCircleFit<3>(tupleMultiplicity_d,
                       3,
                       bField_,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);

    kernelLineFit<3>(tupleMultiplicity_d,
                     3,
                     bField_,
                     outputSoa_d,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     circle_fit_resultsGPU_,
                     offset);

    // quads
    kernelFastFit<4>(
        tuples_d, tupleMultiplicity_d, 4, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);

    kernelCircleFit<4>(tupleMultiplicity_d,
                       4,
                       bField_,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);

    kernelLineFit<4>(tupleMultiplicity_d,
                     4,
                     bField_,
                     outputSoa_d,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     circle_fit_resultsGPU_,
                     offset);

    if (fit5as4_) {
      // penta
      kernelFastFit<4>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);

      kernelCircleFit<4>(tupleMultiplicity_d,
                         5,
                         bField_,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         circle_fit_resultsGPU_,
                         offset);

      kernelLineFit<4>(tupleMultiplicity_d,
                       5,
                       bField_,
                       outputSoa_d,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);

    } else {
      // penta all 5
      kernelFastFit<5>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);

      kernelCircleFit<5>(tupleMultiplicity_d,
                         5,
                         bField_,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         circle_fit_resultsGPU_,
                         offset);

      kernelLineFit<5>(tupleMultiplicity_d,
                       5,
                       bField_,
                       outputSoa_d,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);
    }
  }
}
