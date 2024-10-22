#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

using namespace reco;

RecoChargedCandidate::~RecoChargedCandidate() {}

RecoChargedCandidate *RecoChargedCandidate::clone() const { return new RecoChargedCandidate(*this); }

TrackRef RecoChargedCandidate::track() const { return track_; }

bool RecoChargedCandidate::overlap(const Candidate &c) const {
  const RecoCandidate *o = dynamic_cast<const RecoCandidate *>(&c);
  return (o != nullptr && (checkOverlap(track(), o->track()) || checkOverlap(track(), o->standAloneMuon()) ||
                           checkOverlap(track(), o->combinedMuon())));
}
