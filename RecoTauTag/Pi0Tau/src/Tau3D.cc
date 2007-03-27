#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/Pi0Tau/interface/Tau3D.h"

using namespace reco;

Tau3D::Tau3D(){

  seedTrack_ = reco::TrackRef();
  tracks_.clear();
  pi0s_.clear();

}


Tau3D::Tau3D(const reco::TrackRef seedTrack,
	     const reco::TrackRefVector &trackColl,
	     const reco::Pi0Collection &pi0Coll){

  seedTrack_ = seedTrack;
  tracks_ = trackColl;
  pi0s_ = pi0Coll;

}


Tau3D::Tau3D(const reco::Tau3D& other){

  seedTrack_ = other.seedTrack();
  tracks_ = other.tracks();
  pi0s_ = other.pi0s();

}


std::ostream& reco::operator<<(std::ostream& out, const reco::Tau3D& tau) {  
  if (!out) return out;  

  out << "tracks size : " << tau.tracks().size()
      << ", pi0s size : " << tau.pi0s().size()
      << std::endl;

  return out;
}
