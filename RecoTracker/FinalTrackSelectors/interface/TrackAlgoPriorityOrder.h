#ifndef RecoTracker_FinalTrackSelectors_TrackAlgoPriorityOrder_h
#define RecoTracker_FinalTrackSelectors_TrackAlgoPriorityOrder_h

#include "DataFormats/TrackReco/interface/TrackBase.h"

class TrackAlgoPriorityOrder {
public:
  explicit TrackAlgoPriorityOrder(const std::vector<reco::TrackBase::TrackAlgorithm>& algoOrder);

  unsigned int priority(reco::TrackBase::TrackAlgorithm algo) const { return priority_[algo]; }

private:
  std::array<unsigned int, reco::TrackBase::algoSize> priority_;
};

#endif
