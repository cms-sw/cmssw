// $Id: Photon.cc,v 1.18 2008/10/28 18:51:00 nancy Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::Photon( const LorentzVector & p4, 
		Point caloPos,
		const SuperClusterRef scl,   
		float HoE1, float HoE2, 
		bool hasPixelSeed, 
		const Point & vtx) : 
    RecoCandidate( 0, p4, vtx, 22 ), caloPosition_( caloPos ),
    superCluster_(scl), 
    hadDepth1OverEm_(HoE1), 
    hadDepth2OverEm_(HoE2), 
    pixelSeed_( hasPixelSeed )

{}

Photon::~Photon() { }

Photon * Photon::clone() const { 
  return new Photon( * this ); 
}

reco::SuperClusterRef Photon::superCluster() const {
  return superCluster_;
}



std::vector<reco::ConversionRef>  Photon::conversions() const { 
   return conversions_;
}



bool Photon::hasConversionTracks() const {
  
  if ( this->conversions().size() > 0) 
    return true;
  else
    return false;
}




bool Photon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}

void Photon::setVertex(const Point & vertex) {
  math::XYZVector direction = caloPosition() - vertex;
  double energy = this->energy();
  math::XYZVector momentum = direction.unit() * energy;
  math::XYZTLorentzVector lv(momentum.x(), momentum.y(), momentum.z(), energy );
  setP4(lv);
  vertex_ = vertex;
}


