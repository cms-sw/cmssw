#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::allocateOnGPU(
    Tuples const *tuples,
    caStructures::TupleMultiplicityT<TrackerTraits> const *tupleMultiplicity,
    pixelTrack::TrackSoAT<TrackerTraits> *helix_fit_results) {
  tuples_ = tuples;
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  assert(outputSoa_);
}

template <typename TrackerTraits>
void HelixFitOnGPU<TrackerTraits>::deallocateOnGPU() {}

template class HelixFitOnGPU<pixelTopology::Phase1>;
template class HelixFitOnGPU<pixelTopology::Phase2>;
