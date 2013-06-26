//
// $Id: Photon.cc,v 1.18 2011/06/08 20:40:19 rwolf Exp $
//

#include "DataFormats/PatCandidates/interface/Photon.h"


using pat::Photon;


/// default constructor
Photon::Photon() :
    PATObject<reco::Photon>(reco::Photon()),
    embeddedSuperCluster_(false)
{
}

/// constructor from reco::Photon
Photon::Photon(const reco::Photon & aPhoton) :
    PATObject<reco::Photon>(aPhoton),
    embeddedSuperCluster_(false)
{
}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::RefToBase<reco::Photon> & aPhotonRef) :
    PATObject<reco::Photon>(aPhotonRef),
    embeddedSuperCluster_(false)
{
}

/// constructor from ref to reco::Photon
Photon::Photon(const edm::Ptr<reco::Photon> & aPhotonRef) :
    PATObject<reco::Photon>(aPhotonRef),
    embeddedSuperCluster_(false)
{
}

/// destructor
Photon::~Photon() {
}

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Photon& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Photon: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
  return out; 
}

/// override the superCluster method from CaloJet, to access the internal storage of the supercluster
/// this returns a transient Ref which *should never be persisted*!
reco::SuperClusterRef Photon::superCluster() const {
  if (embeddedSuperCluster_) {
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return reco::Photon::superCluster();
  }
}

/// method to store the photon's supercluster internally
void Photon::embedSuperCluster() {
  superCluster_.clear();
  if (reco::Photon::superCluster().isNonnull()) {
      superCluster_.push_back(*reco::Photon::superCluster());
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
