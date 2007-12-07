// $Id: Photon.cc,v 1.10 2007/10/22 22:24:25 futyand Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::Photon( Charge q, const LorentzVector & p4, Point unconvPos,
		const SuperClusterRef scl,   const ClusterShapeRef shp, 
		bool hasPixelSeed, const Point & vtx) : 
    RecoCandidate( q, p4, vtx, 22 ), unconvPosition_( unconvPos ),
    superCluster_(scl),  seedClusterShape_( shp ),  pixelSeed_( hasPixelSeed ) {
 
}

Photon::~Photon() { }

Photon * Photon::clone() const { 
  return new Photon( * this ); 
}

reco::SuperClusterRef Photon::superCluster() const {
  return superCluster_;
}

reco::ClusterShapeRef Photon::seedClusterShape() const {
  return seedClusterShape_;
}


std::vector<reco::ConversionRef>  Photon::conversions() const { 
   return conversions_;
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
