//
// $Id: Photon.cc,v 1.15 2008/07/08 20:56:48 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/Photon.h"


using pat::Photon;


/// default constructor
Photon::Photon() :
    PATObject<PhotonType>(PhotonType()),
    embeddedSuperCluster_(false)
{
}


/// constructor from PhotonType
Photon::Photon(const PhotonType & aPhoton) :
    PATObject<PhotonType>(aPhoton),
    embeddedSuperCluster_(false)
{
}

/// constructor from ref to PhotonType
Photon::Photon(const edm::RefToBase<PhotonType> & aPhotonRef) :
    PATObject<PhotonType>(aPhotonRef),
    embeddedSuperCluster_(false)
{
}

/// constructor from ref to PhotonType
Photon::Photon(const edm::Ptr<PhotonType> & aPhotonRef) :
    PATObject<PhotonType>(aPhotonRef),
    embeddedSuperCluster_(false)
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

/// method to store the photon's supercluster internally
void Photon::embedSuperCluster() {
  superCluster_.clear();
  if (PhotonType::superCluster().isNonnull()) {
      superCluster_.push_back(*PhotonType::superCluster());
      embeddedSuperCluster_ = true;
  }
}

// method to retrieve a photon ID (or throw)
Bool_t Photon::photonID(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name) return it->second;
  }
  cms::Exception ex("Key not found");
  ex << "pat::Photon: the ID " << name << " can't be found in this pat::Photon.\n";
  ex << "The available IDs are: ";
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    ex << "'" << it->first << "' ";
  }
  ex << ".\n";
  throw ex;
}
// check if an ID is there
bool Photon::isPhotonIDAvailable(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = photonIDs_.begin(), ed = photonIDs_.end(); it != ed; ++it) {
    if (it->first == name) return true;
  }
  return false;
}
