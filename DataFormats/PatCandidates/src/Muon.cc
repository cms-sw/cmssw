//
// $Id$
//

#include "DataFormats/PatCandidates/interface/Muon.h"


using namespace pat;


/// default constructor
Muon::Muon() :
    Lepton<MuonType>(),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false)
{
}


/// constructor from MuonType
Muon::Muon(const MuonType & aMuon) :
    Lepton<MuonType>(aMuon),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false)
{
}


/// constructor from ref to MuonType
Muon::Muon(const edm::RefToBase<MuonType> & aMuonRef) :
    Lepton<MuonType>(aMuonRef),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false)
{
}


/// destructor
Muon::~Muon() {
}


/// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
reco::TrackRef Muon::track() const {
  if (embeddedTrack_) {
    return reco::TrackRef(&track_, 0);
  } else {
    return MuonType::track();
  }
}


/// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
reco::TrackRef Muon::standAloneMuon() const {
  if (embeddedStandAloneMuon_) {
    return reco::TrackRef(&standAloneMuon_, 0);
  } else {
    return MuonType::standAloneMuon();
  }
}


/// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
reco::TrackRef Muon::combinedMuon() const {
  if (embeddedCombinedMuon_) {
    return reco::TrackRef(&combinedMuon_, 0);
  } else {
    return MuonType::combinedMuon();
  }
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


/// embed the Track reconstructed in the tracker only
void Muon::embedTrack() {
  track_.clear();
  track_.push_back(*MuonType::track());
  embeddedTrack_ = true;
}


/// embed the Track reconstructed in the muon detector only
void Muon::embedStandAloneMuon() {
  standAloneMuon_.clear();
  standAloneMuon_.push_back(*MuonType::standAloneMuon());
  embeddedStandAloneMuon_ = true;
}


/// embed the Track reconstructed in both tracked and muon detector
void Muon::embedCombinedMuon() {
  combinedMuon_.clear();
  combinedMuon_.push_back(*MuonType::combinedMuon());
  embeddedCombinedMuon_ = true;
}


/// method to set the lepton ID discriminator
void Muon::setLeptonID(float id) {
  leptonID_ = id;
}
