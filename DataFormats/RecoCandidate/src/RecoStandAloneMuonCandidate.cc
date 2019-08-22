#include "DataFormats/RecoCandidate/interface/RecoStandAloneMuonCandidate.h"

using namespace reco;

RecoStandAloneMuonCandidate::~RecoStandAloneMuonCandidate() {}

RecoStandAloneMuonCandidate *RecoStandAloneMuonCandidate::clone() const {
  return new RecoStandAloneMuonCandidate(*this);
}

TrackRef RecoStandAloneMuonCandidate::standAloneMuon() const { return standAloneMuonTrack_; }

bool RecoStandAloneMuonCandidate::overlap(const Candidate &c) const {
  const RecoCandidate *o = dynamic_cast<const RecoCandidate *>(&c);
  return (o != nullptr &&
          (checkOverlap(standAloneMuon(), o->track()) || checkOverlap(standAloneMuon(), o->standAloneMuon()) ||
           checkOverlap(standAloneMuon(), o->combinedMuon())));
}
