#include "L1Trigger/VertexFinder/interface/RecoVertex.h"

namespace l1tVertexFinder {

  // Template specializations
  template <>
  void RecoVertexWithTP::clear() {
    tracks_.clear();
    trueTracks_.clear();
  }

  template <>
  void RecoVertexWithTP::insert(const L1TrackTruthMatched* fitTrack) {
    tracks_.push_back(fitTrack);
    if (fitTrack->getMatchedTP() != nullptr and fitTrack->getMatchedTP()->physicsCollision())
      trueTracks_.insert(fitTrack->getMatchedTP());
  }

}  // namespace l1tVertexFinder
