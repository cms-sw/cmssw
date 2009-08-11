//
// $Id: Electron.cc,v 1.16 2009/07/08 08:46:38 salerno Exp $
//

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace pat;


/// default constructor
Electron::Electron() :
    Lepton<reco::GsfElectron>(),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    cachedDB_(false),
    dB_(0.0)
{
}


/// constructor from reco::GsfElectron
Electron::Electron(const reco::GsfElectron & anElectron) :
    Lepton<reco::GsfElectron>(anElectron),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    cachedDB_(false),
    dB_(0.0)
{
}


/// constructor from ref to reco::GsfElectron
Electron::Electron(const edm::RefToBase<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    cachedDB_(false),
    dB_(0.0)
{
}

/// constructor from Ptr to reco::GsfElectron
Electron::Electron(const edm::Ptr<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    cachedDB_(false),
    dB_(0.0)
{
}


/// destructor
Electron::~Electron() {
}


/// override the reco::GsfElectron::gsfTrack method, to access the internal storage of the supercluster
reco::GsfTrackRef Electron::gsfTrack() const {
  if (embeddedGsfTrack_) {
    return reco::GsfTrackRef(&gsfTrack_, 0);
  } else {
    return reco::GsfElectron::gsfTrack();
  }
}


/// override the reco::GsfElectron::superCluster method, to access the internal storage of the supercluster
reco::SuperClusterRef Electron::superCluster() const {
  if (embeddedSuperCluster_) {
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return reco::GsfElectron::superCluster();
  }
}


/// override the reco::GsfElectron::track method, to access the internal storage of the track
reco::TrackRef Electron::track() const {
  if (embeddedTrack_) {
    return reco::TrackRef(&track_, 0);
  } else {
    return reco::GsfElectron::track();
  }
}

/// method to store the electron's gsfTrack internally
void Electron::embedGsfTrack() {
  gsfTrack_.clear();
  if (reco::GsfElectron::gsfTrack().isNonnull()) {
      gsfTrack_.push_back(*reco::GsfElectron::gsfTrack());
      embeddedGsfTrack_ = true;
  }
}


/// method to store the electron's supercluster internally
void Electron::embedSuperCluster() {
  superCluster_.clear();
  if (reco::GsfElectron::superCluster().isNonnull()) {
      superCluster_.push_back(*reco::GsfElectron::superCluster());
      embeddedSuperCluster_ = true;
  }
}


/// method to store the electron's track internally
void Electron::embedTrack() {
  track_.clear();
  if (reco::GsfElectron::track().isNonnull()) {
      track_.push_back(*reco::GsfElectron::track());
      embeddedTrack_ = true;
  }
}

// method to retrieve a lepton ID (or throw)
float Electron::electronID(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return it->second;
    }
    cms::Exception ex("Key not found");
    ex << "pat::Electron: the ID " << name << " can't be found in this pat::Electron.\n";
    ex << "The available IDs are: ";
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        ex << "'" << it->first << "' ";
    }
    ex << ".\n";
    throw ex;
}
// check if an ID is there
bool Electron::isElectronIDAvailable(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return true;
    }
    return false;
}

/// reference to the source PFCandidates
reco::PFCandidateRef Electron::pfCandidateRef() const {
  if (embeddedPFCandidate_) {
    return reco::PFCandidateRef(&pfCandidate_, 0);
  } else {
    return pfCandidateRef_;
  }
}
/// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
void Electron::embedPFCandidate() {
  pfCandidate_.clear();
  if ( pfCandidateRef_.isAvailable() && pfCandidateRef_.isNonnull()) {
    pfCandidate_.push_back( *pfCandidateRef_ );
    embeddedPFCandidate_ = true;
  }
}


/// dB gives the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline. 
double Electron::dB() const {
  if ( cachedDB_ ) {
    return dB_;
  } else {
    throw cms::Exception("DataNotFound") << "dB is not stored in this electron object";
    return 0.0;
  }
}
