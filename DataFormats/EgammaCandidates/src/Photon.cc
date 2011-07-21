// $Id: Photon.cc,v 1.25 2011/07/19 16:23:07 nancy Exp $
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
 {}


Photon::Photon( const Photon& rhs ) :
  RecoCandidate(rhs),
  caloPosition_(rhs.caloPosition_),
  photonCore_ ( rhs.photonCore_),
  pixelSeed_  ( rhs.pixelSeed_ ),
  fiducialFlagBlock_ ( rhs.fiducialFlagBlock_ ),
  isolationR04_ ( rhs.isolationR04_),
  isolationR03_ ( rhs.isolationR03_),
  showerShapeBlock_ ( rhs.showerShapeBlock_),
  mipVariableBlock_ (rhs.mipVariableBlock_),
  pfIsolation_ ( rhs.pfIsolation_ )
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
  math::XYZVectorF direction = caloPosition() - vertex;
  double energy = this->energy();
  math::XYZVectorF momentum = direction.unit() * energy;
  math::XYZTLorentzVector lv(momentum.x(), momentum.y(), momentum.z(), energy );
  setP4(lv);
  vertex_ = vertex;
}

reco::SuperClusterRef Photon::superCluster() const {
  return this->photonCore()->superCluster();
}

int Photon::conversionTrackProvenance(const edm::RefToBase<reco::Track>& convTrack) const{

  const reco::ConversionRefVector & conv2leg = this->photonCore()->conversions();
  const reco::ConversionRefVector & conv1leg = this->photonCore()->conversionsOneLeg();

  int origin = -1;
  bool isEg=false, isPf=false;

  for (unsigned iConv=0; iConv<conv2leg.size(); iConv++){
    std::vector<edm::RefToBase<reco::Track> > convtracks = conv2leg[iConv]->tracks();
    for (unsigned itk=0; itk<convtracks.size(); itk++){
      if (convTrack==convtracks[itk]) isEg=true;
    }
  }

  for (unsigned iConv=0; iConv<conv1leg.size(); iConv++){
    std::vector<edm::RefToBase<reco::Track> > convtracks = conv1leg[iConv]->tracks();
    for (unsigned itk=0; itk<convtracks.size(); itk++){
      if (convTrack==convtracks[itk]) isPf=true;
    }
  }

  if (isEg) origin=egamma;
  if (isPf) origin=pflow;
  if (isEg && isPf) origin=both;

  return origin;
}
