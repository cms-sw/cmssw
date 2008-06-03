//
// $Id: Electron.cc,v 1.6.2.1 2008/06/03 20:08:24 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/Electron.h"


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


/// return the lepton ID discriminator
float Electron::leptonID() const {
  return leptonID_;
}


/// return the "robust cuts-based" electron id
float Electron::electronIDRobust() const {
  return electronIDRobust_;
}


/// method to store the electron's gsfTrack internally
void Electron::embedGsfTrack() {
  gsfTrack_.clear();
  gsfTrack_.push_back(*ElectronType::gsfTrack());
  embeddedGsfTrack_ = true;
}


/// method to store the electron's supercluster internally
void Electron::embedSuperCluster() {
  superCluster_.clear();
  superCluster_.push_back(*ElectronType::superCluster());
  embeddedSuperCluster_ = true;
}


/// method to store the electron's track internally
void Electron::embedTrack() {
  track_.clear();
  track_.push_back(*ElectronType::track());
  embeddedTrack_ = true;
}


/// method to set the lepton ID discriminator
void Electron::setLeptonID(float id) {
  leptonID_ = id;
}


/// method to set the "robust cuts-based" electron id
void Electron::setElectronIDRobust(float id) {
  electronIDRobust_ = id;
}

