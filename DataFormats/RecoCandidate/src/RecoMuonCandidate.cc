// $Id: RecoMuonCandidate.cc,v 1.2 2006/04/03 09:05:33 llista Exp $
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
  return muon_->standAlone();
}

TrackRef RecoMuonCandidate::combinedMuon() const {
  return muon_->combined();
}

SuperClusterRef RecoMuonCandidate::superCluster() const {
  return muon_->superCluster();
}
