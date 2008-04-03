//
// $Id$
//

#include "DataFormats/PatCandidates/interface/Photon.h"


using pat::Photon;


/// default constructor
Photon::Photon() :
    PATObject<PhotonType>(PhotonType(reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0, 0, 0), 
				     reco::SuperClusterRef(), reco::ClusterShapeRef(), 0)),
    embeddedSuperCluster_(false),
    photonID_(-1.0) 
{
}


/// constructor from PhotonType
Photon::Photon(const PhotonType & aPhoton) :
    PATObject<PhotonType>(aPhoton),
    embeddedSuperCluster_(false),
    photonID_(-1.0) 
{
}


/// constructor from ref to PhotonType
Photon::Photon(const edm::RefToBase<PhotonType> & aPhotonRef) :
    PATObject<PhotonType>(aPhotonRef),
    embeddedSuperCluster_(false),
    photonID_(-1.0) 
{
}


/// destructor
Photon::~Photon() {
}


/// override the superCluster method from CaloJet, to access the internal storage of the supercluster
/// this returns a transient Ref which *should never be persisted*!
reco::SuperClusterRef Photon::superCluster() const {
  if (embeddedSuperCluster_) {
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return PhotonType::superCluster();
  }
}


/// return the match to the generated photon
const reco::Particle * Photon::genPhoton() const {
  return (genPhoton_.size() > 0 ? &genPhoton_.front() : 0);
}


/// method to store the photon's supercluster internally
void Photon::setSuperCluster(const reco::SuperClusterRef & superCluster) {
  superCluster_.clear();
  superCluster_.push_back(*superCluster);
}


/// method to set the generated photon
void Photon::setGenPhoton(const reco::Particle & gp) {
  genPhoton_.clear();
  genPhoton_.push_back(gp);
}
