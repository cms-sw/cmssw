//
// $Id: Muon.cc,v 1.3 2008/01/22 21:58:15 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Muon.h"


using namespace pat;


/// default constructor
Muon::Muon() : Lepton<MuonType>() {
}


/// constructor from MuonType
Muon::Muon(const MuonType & aMuon) : Lepton<MuonType>(aMuon) {
}


/// constructor from ref to MuonType
Muon::Muon(const edm::RefToBase<MuonType> & aMuonRef) : Lepton<MuonType>(aMuonRef) {
}


/// destructor
Muon::~Muon() {
}


/// return the tracker isolation variable
float Muon::trackIso() const {
  return trackIso_;
}


/// return the calorimeter isolation variable
float Muon::caloIso() const {
  return caloIso_;
}


/// return the lepton ID discriminator
float Muon::leptonID() const {
  return leptonID_;
}


/// method to set the tracker isolation variable
void Muon::setTrackIso(float trackIso) {
  trackIso_ = trackIso;
}


/// method to set the calorimeter isolation variable
void Muon::setCaloIso(float caloIso) {
  caloIso_ = caloIso;
}


/// method to set the lepton ID discriminator
void Muon::setLeptonID(float id) {
  leptonID_ = id;
}
