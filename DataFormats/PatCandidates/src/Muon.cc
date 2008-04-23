//
// $Id: Muon.cc,v 1.5 2008/04/03 12:29:09 gpetrucc Exp $
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


/// return the lepton ID discriminator
float Muon::leptonID() const {
  return leptonID_;
}


/// return the muon segment compatibility -> meant for
float Muon::segmentCompatibility() const {
  return muonid::getSegmentCompatibility(*this);
}


/// return whether it is a good muon
bool Muon::isGoodMuon(const MuonType & muon, muonid::SelectionType type) {
  return muonid::isGoodMuon(*this);
}


/// method to set the lepton ID discriminator
void Muon::setLeptonID(float id) {
  leptonID_ = id;
}
