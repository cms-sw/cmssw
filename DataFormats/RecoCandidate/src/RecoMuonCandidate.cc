// $Id: RecoMuonCandidate.cc,v 1.4 2006/02/23 16:52:39 llista Exp $
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
