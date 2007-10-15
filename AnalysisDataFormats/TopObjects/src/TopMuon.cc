//
// $Id: TopMuon.cc,v 1.1 2007/09/20 18:12:22 lowette Exp $
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
  return trackIso_;
}


/// return the calorimeter isolation variable
double TopMuon::getCaloIso() const {
  return caloIso_;
}


/// return the lepton ID discriminator
double TopMuon::getLeptonID() const {
  return leptonID_;
}


/// method to set the tracker isolation variable
void TopMuon::setTrackIso(double trackIso) {
  trackIso_ = trackIso;
}


/// method to set the calorimeter isolation variable
void TopMuon::setCaloIso(double caloIso) {
  caloIso_ = caloIso;
}


/// method to set the lepton ID discriminator
void TopMuon::setLeptonID(double id) {
  leptonID_ = id;
}
