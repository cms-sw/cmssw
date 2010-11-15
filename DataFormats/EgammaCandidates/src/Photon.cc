// $Id: Photon.cc,v 1.20 2009/03/24 18:03:39 nancy Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::Photon( const LorentzVector & p4, 
		Point caloPos,
                const PhotonCoreRef & core,
		const Point & vtx) : 
    RecoCandidate( 0, p4, vtx, 22 ), 
    caloPosition_( caloPos ),
    photonCore_(core),
    pixelSeed_(false)
{ }


Photon::Photon( const Photon& rhs ) : 
  RecoCandidate(rhs),
  caloPosition_(rhs.caloPosition_), 
  photonCore_ ( rhs.photonCore_),
  pixelSeed_  ( rhs.pixelSeed_ ),
  fiducialFlagBlock_ ( rhs.fiducialFlagBlock_ ),
  isolationR04_ ( rhs.isolationR04_),
  isolationR03_ ( rhs.isolationR03_),
  showerShapeBlock_ ( rhs.showerShapeBlock_)
{}


 

Photon::~Photon() { }

Photon * Photon::clone() const { 
  return new Photon( * this ); 
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


