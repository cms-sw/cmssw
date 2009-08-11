//
// $Id: Muon.cc,v 1.19 2009/06/22 15:58:32 jribnik Exp $
//

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace pat;


/// default constructor
Muon::Muon() :
    Lepton<reco::Muon>(),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    cachedDB_(false),
    cachedNumberOfValidHits_(0),
    normChi2_(0.0),
    dB_(0.0),
    numberOfValidHits_(0)
{
}


/// constructor from reco::Muon
Muon::Muon(const reco::Muon & aMuon) :
    Lepton<reco::Muon>(aMuon),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    cachedDB_(false),
    cachedNumberOfValidHits_(0),
    normChi2_(0.0),
    dB_(0.0),
    numberOfValidHits_(0)
{
}


/// constructor from ref to reco::Muon
Muon::Muon(const edm::RefToBase<reco::Muon> & aMuonRef) :
    Lepton<reco::Muon>(aMuonRef),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    cachedDB_(false),
    cachedNumberOfValidHits_(0),
    normChi2_(0.0),
    dB_(0.0),
    numberOfValidHits_(0)
{
}


/// constructor from ref to reco::Muon
Muon::Muon(const edm::Ptr<reco::Muon> & aMuonRef) :
    Lepton<reco::Muon>(aMuonRef),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    pickyMuonRef_(),
    tpfmsMuonRef_(),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    cachedDB_(false),
    cachedNumberOfValidHits_(0),
    normChi2_(0.0),
    dB_(0.0),
    numberOfValidHits_(0)
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
    return reco::Muon::innerTrack();
  }
}


/// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
reco::TrackRef Muon::standAloneMuon() const {
  if (embeddedStandAloneMuon_) {
    return reco::TrackRef(&standAloneMuon_, 0);
  } else {
    return reco::Muon::outerTrack();
  }
}


/// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
reco::TrackRef Muon::combinedMuon() const {
  if (embeddedCombinedMuon_) {
    return reco::TrackRef(&combinedMuon_, 0);
  } else {
    return reco::Muon::globalTrack();
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


/// embed the Track reconstructed in the tracker only
void Muon::embedTrack() {
  track_.clear();
  if (reco::Muon::innerTrack().isNonnull()) {
      track_.push_back(*reco::Muon::innerTrack());
      embeddedTrack_ = true;
  }
}


/// embed the Track reconstructed in the muon detector only
void Muon::embedStandAloneMuon() {
  standAloneMuon_.clear();
  if (reco::Muon::outerTrack().isNonnull()) {
      standAloneMuon_.push_back(*reco::Muon::outerTrack());
      embeddedStandAloneMuon_ = true;
  }
}


/// embed the Track reconstructed in both tracked and muon detector
void Muon::embedCombinedMuon() {
  combinedMuon_.clear();
  if (reco::Muon::globalTrack().isNonnull()) {
      combinedMuon_.push_back(*reco::Muon::globalTrack());
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

/// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
void Muon::embedPFCandidate() {
  pfCandidate_.clear();
  if ( pfCandidateRef_.isAvailable() && pfCandidateRef_.isNonnull()) {
    pfCandidate_.push_back( *pfCandidateRef_ );
    embeddedPFCandidate_ = true;
  }
}

bool Muon::muonID(const std::string& name) const {
  muon::SelectionType st = muon::selectionTypeFromString(name);
  return muon::isGoodMuon(*this, st);
}


/// Norm chi2 gives the normalized chi2 of the global track. 
/// The user can choose to cache this info so they can drop the
/// global track, or they can use the track itself if it is present
/// in the event. 
double Muon::normChi2() const {
  if ( cachedNormChi2_ ) {
    return normChi2_;
  } else {
    reco::TrackRef t = this->globalTrack();
    if ( t.isNonnull() && t.isAvailable() ) {
      return t->chi2() / t->ndof();
    }
    else {
      throw cms::Exception("DataNotFound") << "Track ref is null in accessing normChi2";
      return 0.0;
    }
  }
}

/// numberOfValidHits returns the number of valid hits on the global track.
/// The user can choose to cache this info so they can drop the
/// global track, or they can use the track itself if it is present
/// in the event. 
unsigned int Muon::numberOfValidHits() const {
  if ( cachedNumberOfValidHits_ ) {
    return numberOfValidHits_;
  } else {
    reco::TrackRef t = this->globalTrack();
    if ( t.isNonnull() && t.isAvailable() ) {
      return t->numberOfValidHits();
    }
    else {
      throw cms::Exception("DataNotFound") << "Track ref is null in accessing normChi2";
      return 0;
    }
  }
}

/// dB gives the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline. 
double Muon::dB() const {
  if ( cachedDB_ ) {
    return dB_;
  } else {
    throw cms::Exception("DataNotFound") << "Track ref is null in accessing normChi2";
    return 0.0;
  }
}
