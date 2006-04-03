// $Id: RecoMuonCandidate.cc,v 1.1 2006/02/28 10:59:16 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoMuonCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"

using namespace reco;

RecoMuonCandidate::~RecoMuonCandidate() { }

RecoMuonCandidate * RecoMuonCandidate::clone() const { 
  return new RecoMuonCandidate( * this ); 
}

MuonRef RecoMuonCandidate::muon() const {
  return muon_;
}

TrackRef RecoMuonCandidate::track() const {
  return muon_->track();
}

TrackRef RecoMuonCandidate::standAloneMuon() const {
  return muon_->standAloneMuon();
}
