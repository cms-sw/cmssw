//
// $Id: Electron.cc,v 1.4 2008/01/26 20:19:45 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/Electron.h"


using namespace pat;


/// default constructor
Electron::Electron() : Lepton<ElectronType>() {
}


/// constructor from ElectronType
Electron::Electron(const ElectronType & anElectron) : Lepton<ElectronType>(anElectron) {
}


/// constructor from ref to ElectronType
Electron::Electron(const edm::RefToBase<ElectronType> & anElectronRef) : Lepton<ElectronType>(anElectronRef) {
}


/// destructor
Electron::~Electron() {
}

/// return the lepton ID discriminator
float Electron::leptonID() const {
  return leptonID_;
}


/// return the "robust cuts-based" electron id
float Electron::electronIDRobust() const {
  return electronIDRobust_;
}


/// method to set the lepton ID discriminator
void Electron::setLeptonID(float id) {
  leptonID_ = id;
}


/// method to set the "robust cuts-based" electron id
void Electron::setElectronIDRobust(float id) {
  electronIDRobust_ = id;
}

