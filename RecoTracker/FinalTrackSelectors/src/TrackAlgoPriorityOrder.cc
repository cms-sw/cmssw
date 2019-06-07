#include "RecoTracker/FinalTrackSelectors/interface/TrackAlgoPriorityOrder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "trackAlgoPriorityOrder.h"

TrackAlgoPriorityOrder::TrackAlgoPriorityOrder(const std::vector<reco::TrackBase::TrackAlgorithm>& algoOrder)
    : priority_(trackAlgoPriorityOrder) {
  // with less than 1 element there is nothing to do
  if (algoOrder.size() <= 1)
    return;

  // Reordering the algo priorities is just a matter of taking the
  // current priorities of the algos, sorting them, and inserting back
  //
  // iter0  2     2
  // iter1  4  -> 3
  // iter2  3     4
  std::vector<unsigned int> priorities;
  priorities.reserve(algoOrder.size());
  for (const auto algo : algoOrder) {
    priorities.push_back(trackAlgoPriorityOrder[algo]);
  }

  std::sort(priorities.begin(), priorities.end());

  for (size_t i = 0, end = priorities.size(); i != end; ++i) {
    priority_[algoOrder[i]] = priorities[i];
  }
}
