//
// $Id: Photon.cc,v 1.14 2008/06/23 22:22:18 gpetrucc Exp $
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

