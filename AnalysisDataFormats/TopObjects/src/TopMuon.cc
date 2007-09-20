//
// $Id$
//

#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"


/// default constructor
TopMuon::TopMuon() : TopLepton<TopMuonType>() {
}


/// constructor from TopMuonType
TopMuon::TopMuon(const TopMuonType & aMuon) : TopLepton<TopMuonType>(aMuon) {
}


/// destructor
TopMuon::~TopMuon() {
}


/// return the tracker isolation variable
double TopMuon::getTrackIso() const {
  return this->getIsolationR03().sumPt;
}


/// return the calorimeter isolation variable
double TopMuon::getCaloIso() const {
  return this->getIsolationR03().emEt + this->getIsolationR03().hadEt + this->getIsolationR03().hoEt;
}


/// return the lepton ID discriminator
double TopMuon::getLeptonID() const {
  return leptonID_;
}


/// method to set the lepton ID discriminator
void TopMuon::setLeptonID(double id) {
  leptonID_ = id;
}
