// $Id: Photon.cc,v 1.9 2007/10/21 22:01:59 futyand Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 
#include "DataFormats/EgammaReco/interface/ClusterShape.h"

using namespace reco;

Photon::Photon( Charge q, const LorentzVector & p4, Point unconvPos,
		const SuperClusterRef scl, const ClusterShapeRef shp,
		bool hasPixelSeed, const Point & vtx) : 
  RecoCandidate( q, p4, vtx, 22 ), unconvPosition_( unconvPos ),
  superCluster_(scl), seedClusterShape_( shp ), pixelSeed_( hasPixelSeed ) {

  // compute R9=E3x3/ESC
  r9_ = seedClusterShape_->e3x3()/(superCluster_->rawEnergy()+superCluster_->preshowerEnergy());
  r19_ = seedClusterShape_->eMax()/seedClusterShape_->e3x3();
  e5x5_ = seedClusterShape_->e5x5();
  
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
  if (r9_>0.93) {
    return unconvPosition_;
  } else {
    return superCluster()->position();
  }
}
