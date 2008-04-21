// $Id: Photon.cc,v 1.14 2008/03/03 20:34:38 nancy Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::Photon( const LorentzVector & p4, Point unconvPos,
		const SuperClusterRef scl,   
		float HoE, 
		float r9,
		float r19,
	        float e5x5,
		bool hasPixelSeed, 
		const Point & vtx) : 
    RecoCandidate( 0, p4, vtx, 22 ), unconvPosition_( unconvPos ),
    superCluster_(scl), 
    hadOverEm_(HoE), 
    r9_(r9),
    r19_(r19),
    e5x5_(e5x5),
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

math::XYZPoint Photon::caloPosition() const {
  if (this->r9()>0.93) {
    return unconvPosition_;
  } else {
    return superCluster()->position();
  }
}
