//
// $Id: Electron.cc,v 1.8 2008/06/13 09:55:35 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace pat;


/// default constructor
Electron::Electron() :
    Lepton<ElectronType>(),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false)
{
}


/// constructor from ElectronType
Electron::Electron(const ElectronType & anElectron) :
    Lepton<ElectronType>(anElectron),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false)
{
}


/// constructor from ref to ElectronType
Electron::Electron(const edm::RefToBase<ElectronType> & anElectronRef) :
    Lepton<ElectronType>(anElectronRef),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false)
{
}

/// constructor from ref to ElectronType
Electron::Electron(const edm::Ptr<ElectronType> & anElectronRef) :
    Lepton<ElectronType>(anElectronRef),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false)
{
}


/// destructor
Electron::~Electron() {
}


/// override the ElectronType::gsfTrack method, to access the internal storage of the supercluster
reco::GsfTrackRef Electron::gsfTrack() const {
  if (embeddedGsfTrack_) {
    return reco::GsfTrackRef(&gsfTrack_, 0);
  } else {
    return ElectronType::gsfTrack();
  }
}


/// override the ElectronType::superCluster method, to access the internal storage of the supercluster
reco::SuperClusterRef Electron::superCluster() const {
  if (embeddedSuperCluster_) {
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return ElectronType::superCluster();
  }
}


/// override the ElectronType::track method, to access the internal storage of the track
reco::TrackRef Electron::track() const {
  if (embeddedTrack_) {
    return reco::TrackRef(&track_, 0);
  } else {
    return ElectronType::track();
  }
}

/// method to store the electron's gsfTrack internally
void Electron::embedGsfTrack() {
  gsfTrack_.clear();
  if (ElectronType::gsfTrack().isNonnull()) {
      gsfTrack_.push_back(*ElectronType::gsfTrack());
      embeddedGsfTrack_ = true;
  }
}


/// method to store the electron's supercluster internally
void Electron::embedSuperCluster() {
  superCluster_.clear();
  if (ElectronType::superCluster().isNonnull()) {
      superCluster_.push_back(*ElectronType::superCluster());
      embeddedSuperCluster_ = true;
  }
}


/// method to store the electron's track internally
void Electron::embedTrack() {
  track_.clear();
  if (ElectronType::track().isNonnull()) {
      track_.push_back(*ElectronType::track());
      embeddedTrack_ = true;
  }
}

#ifdef PAT_patElectron_Default_eID
/// method to retrieve default leptonID (or throw)
float Electron::leptonID() const { 
    if (leptonIDs_.empty()) {
     #ifdef PAT_patElectron_eID_Throw
        throw cms::Exception("No data") << "This pat::Electron does not contain any ID.\n";
     #else
        return -1.0
     #endif
    }
    return leptonIDs_.front().second; 
}
// Return the name of the default lepton id name, or "" if none was configured
const std::string & Electron::leptonIDname() const { 
    static std::string NONE = "NULL";
    return leptonIDs_.empty() ? NONE : leptonIDs_.front().first; 
}
#endif

// method to retrieve a lepton ID (or throw)
float Electron::leptonID(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = leptonIDs_.begin(), ed = leptonIDs_.end(); it != ed; ++it) {
        if (it->first == name) return it->second;
    }
#ifdef PAT_patElectron_eID_Throw
    cms::Exception ex("Key not found");
    ex << "pat::Electron: the ID " << name << " can't be found in this pat::Electron.\n";
    ex << "The available IDs are: ";
    for (std::vector<IdPair>::const_iterator it = leptonIDs_.begin(), ed = leptonIDs_.end(); it != ed; ++it) {
        ex << "'" << it->first << "' ";
    }
    ex << ".\n";
    throw ex;
#else
    return -1.0;
#endif
}
// check if an ID is there
bool Electron::isLeptonIDAvailable(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = leptonIDs_.begin(), ed = leptonIDs_.end(); it != ed; ++it) {
        if (it->first == name) return true;
    }
    return false;
}

