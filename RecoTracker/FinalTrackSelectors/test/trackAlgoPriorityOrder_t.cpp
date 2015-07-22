#include "RecoTracker/FinalTrackSelectors/plugins/trackAlgoPriorityOrder.h"

#include <iostream>

int main(void) {
  for(unsigned int ialgo = 0; ialgo < reco::TrackBase::algoSize; ++ialgo) {
    reco::TrackBase::TrackAlgorithm algo = static_cast<reco::TrackBase::TrackAlgorithm>(ialgo);

    const unsigned int priority = trackAlgoPriorityOrder[algo];
    std::cout << "Algorithm " << reco::TrackBase::algoName(algo) << " has priority " << priority << std::endl;

    if(impl::algoPriorityOrder[priority] != algo) {
      std::cout << "Priority for algo " << reco::TrackBase::algoName(algo) << " is inconsistent: algo " << algo << " has priority " << priority << ", which maps back to algo " << impl::algoPriorityOrder[priority] << std::endl;
      return 1;
    }
  }

  return 0;
}
