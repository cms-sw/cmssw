#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernelsOnCPU(HitsView const* hv, uint32_t hitsInFit, uint32_t maxNumberOfTuples) {
  assert(tuples_);

#ifdef BROKENLINE_DEBUG
  setlinebuf(stdout);
#endif

  //  Fit internals
  auto tkidGPU = std::make_unique<caConstants::tindex_type[]>(maxNumberOfConcurrentFits_);
  auto hitsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<6>) / sizeof(double));
  auto hits_geGPU =
      std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<6>) / sizeof(float));
  auto fast_fit_resultsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernel_BLFastFit<3>(tuples_,
                        tupleMultiplicity_,
                        hv,
                        tkidGPU.get(),
                        hitsGPU.get(),
                        hits_geGPU.get(),
                        fast_fit_resultsGPU.get(),
                        3,
                        3,
                        offset);

    kernel_BLFit<3>(tupleMultiplicity_,
                    bField_,
                    outputSoa_,
                    tkidGPU.get(),
                    hitsGPU.get(),
                    hits_geGPU.get(),
                    fast_fit_resultsGPU.get());

    if (fitNas4_) {
      // fit all as 4
      kernel_BLFastFit<4>(tuples_,
                          tupleMultiplicity_,
                          hv,
                          tkidGPU.get(),
                          hitsGPU.get(),
                          hits_geGPU.get(),
                          fast_fit_resultsGPU.get(),
                          4,
                          8,
                          offset);

      kernel_BLFit<4>(tupleMultiplicity_,
                      bField_,
                      outputSoa_,
                      tkidGPU.get(),
                      hitsGPU.get(),
                      hits_geGPU.get(),
                      fast_fit_resultsGPU.get());
    } else {
      // fit quads
      kernel_BLFastFit<4>(tuples_,
                          tupleMultiplicity_,
                          hv,
                          tkidGPU.get(),
                          hitsGPU.get(),
                          hits_geGPU.get(),
                          fast_fit_resultsGPU.get(),
                          4,
                          4,
                          offset);

      kernel_BLFit<4>(tupleMultiplicity_,
                      bField_,
                      outputSoa_,
                      tkidGPU.get(),
                      hitsGPU.get(),
                      hits_geGPU.get(),
                      fast_fit_resultsGPU.get());
      // fit penta (all 5)
      kernel_BLFastFit<5>(tuples_,
                          tupleMultiplicity_,
                          hv,
                          tkidGPU.get(),
                          hitsGPU.get(),
                          hits_geGPU.get(),
                          fast_fit_resultsGPU.get(),
                          5,
                          5,
                          offset);

      kernel_BLFit<5>(tupleMultiplicity_,
                      bField_,
                      outputSoa_,
                      tkidGPU.get(),
                      hitsGPU.get(),
                      hits_geGPU.get(),
                      fast_fit_resultsGPU.get());
      // fit sexta and above (as 6)
      kernel_BLFastFit<6>(tuples_,
                          tupleMultiplicity_,
                          hv,
                          tkidGPU.get(),
                          hitsGPU.get(),
                          hits_geGPU.get(),
                          fast_fit_resultsGPU.get(),
                          6,
                          8,
                          offset);

      kernel_BLFit<6>(tupleMultiplicity_,
                      bField_,
                      outputSoa_,
                      tkidGPU.get(),
                      hitsGPU.get(),
                      hits_geGPU.get(),
                      fast_fit_resultsGPU.get());
    }

  }  // loop on concurrent fits
}
