#include "L1Trigger/VertexFinder/interface/RecoVertex.h"

namespace l1tVertexFinder {

  // Template specializations
  template <>
  RecoVertexWithTP& RecoVertexWithTP::operator+=(const RecoVertex& rhs) {
    this->tracks_.insert(std::end(this->tracks_), std::begin(rhs.tracks()), std::end(rhs.tracks()));
    this->trueTracks_.insert(std::begin(rhs.trueTracks()), std::end(rhs.trueTracks()));
    return *this;
  }

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
