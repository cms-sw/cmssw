#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidate.h"

using namespace reco;

HcalIsolatedTrackCandidate::HcalIsolatedTrackCandidate(const HcalIsolatedTrackCandidate& right) : RecoCandidate(right), track_(right.track_), l1Jet_(right.l1Jet_) {
  maxP_      = right.maxP_;
  enEcal_    = right.enEcal_;
  ptL1_      = right.ptL1_;
  etaL1_     = right.etaL1_; 
  phiL1_     = right.phiL1_;  
  etaPhiEcal_= right.etaPhiEcal_;
  etaEcal_   = right.etaEcal_; 
  phiEcal_   = right.phiEcal_;  
  etaPhiHcal_= right.etaPhiHcal_;
  etaHcal_   = right.etaHcal_; 
  phiHcal_   = right.phiHcal_;  
  ietaHcal_  = right.ietaHcal_; 
  iphiHcal_  = right.iphiHcal_;  
}

HcalIsolatedTrackCandidate::~HcalIsolatedTrackCandidate() { }

HcalIsolatedTrackCandidate * HcalIsolatedTrackCandidate::clone() const { 
  return new HcalIsolatedTrackCandidate( * this ); 
}

reco::TrackRef HcalIsolatedTrackCandidate::track() const {
  return track_;
}

l1extra::L1JetParticleRef HcalIsolatedTrackCandidate::l1jet() const {
  return l1Jet_;
}

math::XYZTLorentzVector HcalIsolatedTrackCandidate::l1jetp() const {
  double pL1_ = ptL1_*cosh(etaL1_);
  math::XYZTLorentzVector pL1(ptL1_*cos(phiL1_), ptL1_*sin(phiL1_),
			      pL1_*tanh(etaL1_), pL1_);
  return pL1;
}

bool HcalIsolatedTrackCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  checkOverlap( track(), o->track() ) );
}
