//
// $Id: Muon.cc,v 1.15.2.2 2009/03/19 17:33:27 lusito Exp $
//

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace pat;


/// default constructor
Muon::Muon() :
    Lepton<MuonType>(),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false)
    //pfCandidateRef_()
{
}


/// constructor from MuonType
Muon::Muon(const MuonType & aMuon) :
    Lepton<MuonType>(aMuon),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false)
    //pfCandidateRef_()
{
}


/// constructor from ref to MuonType
Muon::Muon(const edm::RefToBase<MuonType> & aMuonRef) :
    Lepton<MuonType>(aMuonRef),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false)
    //pfCandidateRef_()
{
}


/// constructor from ref to MuonType
Muon::Muon(const edm::Ptr<MuonType> & aMuonRef) :
    Lepton<MuonType>(aMuonRef),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false)
    //pfCandidateRef_()
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
    return MuonType::innerTrack();
  }
}


/// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
reco::TrackRef Muon::standAloneMuon() const {
  if (embeddedStandAloneMuon_) {
    return reco::TrackRef(&standAloneMuon_, 0);
  } else {
    return MuonType::outerTrack();
  }
}


/// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
reco::TrackRef Muon::combinedMuon() const {
  if (embeddedCombinedMuon_) {
    return reco::TrackRef(&combinedMuon_, 0);
  } else {
    return MuonType::globalTrack();
  }
}

/// reference to Track reconstructed using hits in the tracker + "good" muon hits
reco::TrackRef Muon::pickyMuon() const {
  if (embeddedPickyMuon_) {
    return reco::TrackRef(&pickyMuon_, 0);
  } else {
    return pickyMuonRef_;
  }
}

/// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits
reco::TrackRef Muon::tpfmsMuon() const {
  if (embeddedTpfmsMuon_) {
    return reco::TrackRef(&tpfmsMuon_, 0);
  } else {
    return tpfmsMuonRef_;
  }
}

/// reference to the source IsolatedPFCandidates
reco::PFCandidateRef Muon::pfCandidateRef() const {
  if (embeddedPFCandidate_) {
    return reco::PFCandidateRef(&pfCandidate_, 0);
  } else {
    return pfCandidateRef_;
  }
}

/// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
void Muon::embedPFCandidate() {
  pfCandidate_.clear();
  if ( pfCandidateRef_.isAvailable() && pfCandidateRef_.isNonnull()) {
    pfCandidate_.push_back( *pfCandidateRef_ );
    embeddedPFCandidate_ = true;
  }
}



/// embed the Track reconstructed in the tracker only
void Muon::embedTrack() {
  track_.clear();
  if (MuonType::innerTrack().isNonnull()) {
      track_.push_back(*MuonType::innerTrack());
      embeddedTrack_ = true;
  }
}




/// embed the Track reconstructed in both tracked and muon detector
void Muon::embedCombinedMuon() {
  combinedMuon_.clear();
  if (MuonType::globalTrack().isNonnull()) {
      combinedMuon_.push_back(*MuonType::globalTrack());
      embeddedCombinedMuon_ = true;
  }
}

/// embed the picky Track
void Muon::embedPickyMuon() {
  pickyMuon_.clear();
  if (pickyMuonRef_.isNonnull()) {
      pickyMuon_.push_back(*pickyMuonRef_);
      embeddedPickyMuon_ = true;
  }
}

/// embed the tpfms Track
void Muon::embedTpfmsMuon() {
  tpfmsMuon_.clear();
  if (tpfmsMuonRef_.isNonnull()) {
      tpfmsMuon_.push_back(*tpfmsMuonRef_);
      embeddedTpfmsMuon_ = true;
  }
}


/// embed the Track reconstructed in the muon detector only	 
void Muon::embedStandAloneMuon() {	 
  standAloneMuon_.clear();	 
  if (MuonType::outerTrack().isNonnull()) {	 
      standAloneMuon_.push_back(*MuonType::outerTrack());	 
      embeddedStandAloneMuon_ = true;	 
  }	 
}
