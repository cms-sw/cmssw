// $Id: Photon.cc,v 1.16 2008/04/21 23:16:16 nancy Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::Photon( const LorentzVector & p4, 
		Point caloPos,
		const SuperClusterRef scl,   
		float HoE, 
		bool hasPixelSeed, 
		const Point & vtx) : 
    RecoCandidate( 0, p4, vtx, 22 ), caloPosition_( caloPos ),
    superCluster_(scl), 
    hadOverEm_(HoE), 
    pixelSeed_( hasPixelSeed ) {}

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

bool Photon::isConverted() const {
  
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


