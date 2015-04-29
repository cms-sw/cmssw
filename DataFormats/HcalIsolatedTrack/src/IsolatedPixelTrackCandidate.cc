#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

using namespace reco;

IsolatedPixelTrackCandidate::IsolatedPixelTrackCandidate(const IsolatedPixelTrackCandidate& right) : RecoCandidate(right), track_(right.track_), l1tauJet_(right.l1tauJet_) {
  maxPtPxl_  = right.maxPtPxl_;
  sumPtPxl_  = right.sumPtPxl_;
  enIn_      = right.enIn_;
  enOut_     = right.enOut_;
  nhitIn_    = right.nhitIn_;
  nhitOut_   = right.nhitOut_;
  etaPhiEcal_= right.etaPhiEcal_;
  etaEcal_   = right.etaEcal_; 
  phiEcal_   = right.phiEcal_;  
}

IsolatedPixelTrackCandidate::~IsolatedPixelTrackCandidate() { }


IsolatedPixelTrackCandidate * IsolatedPixelTrackCandidate::clone() const { 
  return new IsolatedPixelTrackCandidate( * this ); 
}

TrackRef IsolatedPixelTrackCandidate::track() const {
  return track_;
}

l1extra::L1JetParticleRef IsolatedPixelTrackCandidate::l1tau() const {
  return l1tauJet_;
}

bool IsolatedPixelTrackCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  checkOverlap( track(), o->track() ) );
}

std::pair<int,int> IsolatedPixelTrackCandidate::towerIndex() const {

  int ieta(0), iphi(0), nphi(72), kphi(1);
  double etas[24]={0.000,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,
		   0.783,0.870,0.957,1.044,1.131,1.218,1.305,1.392,1.479,
		   1.566,1.653,1.740,1.830,1.930,2.043};
  for (int i=1; i<24; i++) {
    if (fabs(track_->eta())<=etas[i]) {
      ieta = (track_->eta() > 0) ? i : -i;
      if (i > 20) {
	kphi = 2; nphi = 36;
      }
      break;
    }
  }

  const double dphi=M_PI/36.; //0.087266462;
  double phi = track_->phi();
  if (phi < 0) phi += (2*M_PI);
  double delta = phi+(kphi*dphi);
  for (int i=0; i<nphi; i++) {
    if (delta<=(kphi*(i+1)*dphi)) {
      iphi = kphi*i + 1;
      break;
    }
  }

  return std::pair<int,int>(ieta,iphi);

}
