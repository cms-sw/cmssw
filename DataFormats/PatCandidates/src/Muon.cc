//
//

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include <limits>

using namespace pat;


/// default constructor
Muon::Muon() :
    Lepton<reco::Muon>(),
    embeddedMuonBestTrack_(false),
    embeddedTunePMuonBestTrack_(false),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedTCMETMuonCorrs_(false),
    embeddedCaloMETMuonCorrs_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    embeddedDytMuon_(false),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    normChi2_(0.0),
    cachedNumberOfValidHits_(false),
    numberOfValidHits_(0),
    pfEcalEnergy_(0)
{
  initImpactParameters();
}

/// constructor from reco::Muon
Muon::Muon(const reco::Muon & aMuon) :
    Lepton<reco::Muon>(aMuon),
    embeddedMuonBestTrack_(false),
    embeddedTunePMuonBestTrack_(false),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedTCMETMuonCorrs_(false),
    embeddedCaloMETMuonCorrs_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    embeddedDytMuon_(false),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    normChi2_(0.0),
    cachedNumberOfValidHits_(false),
    numberOfValidHits_(0),
    pfEcalEnergy_(0)
{
  initImpactParameters();
}

/// constructor from ref to reco::Muon
Muon::Muon(const edm::RefToBase<reco::Muon> & aMuonRef) :
    Lepton<reco::Muon>(aMuonRef),
    embeddedMuonBestTrack_(false),
    embeddedTunePMuonBestTrack_(false),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedTCMETMuonCorrs_(false),
    embeddedCaloMETMuonCorrs_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    embeddedDytMuon_(false),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    normChi2_(0.0),
    cachedNumberOfValidHits_(0),
    numberOfValidHits_(0),
    pfEcalEnergy_(0)
{
  initImpactParameters();
}

/// constructor from ref to reco::Muon
Muon::Muon(const edm::Ptr<reco::Muon> & aMuonRef) :
    Lepton<reco::Muon>(aMuonRef),
    embeddedMuonBestTrack_(false),
    embeddedTunePMuonBestTrack_(false),
    embeddedTrack_(false),
    embeddedStandAloneMuon_(false),
    embeddedCombinedMuon_(false),
    embeddedTCMETMuonCorrs_(false),
    embeddedCaloMETMuonCorrs_(false),
    embeddedPickyMuon_(false),
    embeddedTpfmsMuon_(false),
    embeddedDytMuon_(false),
    embeddedPFCandidate_(false),
    pfCandidateRef_(),
    cachedNormChi2_(false),
    normChi2_(0.0),
    cachedNumberOfValidHits_(0),
    numberOfValidHits_(0),
    pfEcalEnergy_(0)
{
  initImpactParameters();
}

/// destructor
Muon::~Muon() {
}

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Muon& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Muon: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
  return out; 
}

// initialize impact parameter container vars
void Muon::initImpactParameters() {
  std::fill(ip_, ip_+IpTypeSize, 0.0f);
  std::fill(eip_, eip_+IpTypeSize, 0.0f);
  cachedIP_ = 0;
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
reco::TrackRef Muon::pickyTrack() const {
  if (embeddedPickyMuon_) {
    return reco::TrackRef(&pickyMuon_, 0);
  } else {
    return reco::Muon::pickyTrack();
  }
}

/// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits
reco::TrackRef Muon::tpfmsTrack() const {
  if (embeddedTpfmsMuon_) {
    return reco::TrackRef(&tpfmsMuon_, 0);
  } else {
    return reco::Muon::tpfmsTrack();
  }
}

/// reference to Track reconstructed using hits in the tracker + info from the first muon station that has hits
reco::TrackRef Muon::dytTrack() const {
  if (embeddedDytMuon_) {
    return reco::TrackRef(&dytMuon_, 0);
  } else {
    return reco::Muon::dytTrack();
  }
}

/// reference to Track giving best momentum (global PFlow algo) 
reco::TrackRef Muon::muonBestTrack() const {
  if (!muonBestTrack_.empty()) {
    return reco::TrackRef(&muonBestTrack_, 0);
  } else {
    return reco::Muon::muonBestTrack();
  }
}

/// reference to Track giving best momentum (muon only) 
reco::TrackRef Muon::tunePMuonBestTrack() const {
  if (!tunePMuonBestTrack_.empty()) {
    return reco::TrackRef(&tunePMuonBestTrack_, 0);
  } else if (muonBestTrackType() == tunePMuonBestTrackType()) {
    return muonBestTrack();
  } else {
    return reco::Muon::tunePMuonBestTrack();
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

/// reference to the parent PF candidate for use in TopProjector
reco::CandidatePtr Muon::sourceCandidatePtr( size_type i ) const {
  if(pfCandidateRef_.isNonnull() && i==0 ) return reco::CandidatePtr(edm::refToPtr(pfCandidateRef_) );
  if(refToOrig_.isNonnull() &&  pfCandidateRef_.isNonnull() && i==1 ) return refToOrig_;
  if(refToOrig_.isNonnull() && ! pfCandidateRef_.isNonnull() && i==0 ) return refToOrig_;
  return reco::CandidatePtr();
}

/// embed the Track selected to be the best measurement of the muon parameters
void Muon::embedMuonBestTrack(bool force) {
  muonBestTrack_.clear();
  embeddedMuonBestTrack_ = false;
  bool alreadyEmbedded = false;
  if (!force) {
      switch (muonBestTrackType()) {
        case None: alreadyEmbedded = true; break;
        case InnerTrack: if (embeddedTrack_) alreadyEmbedded = true; break;
        case OuterTrack: if (embeddedStandAloneMuon_) alreadyEmbedded = true; break;
        case CombinedTrack: if (embeddedCombinedMuon_) alreadyEmbedded = true; break;
        case TPFMS: if (embeddedTpfmsMuon_) alreadyEmbedded = true; break;
        case Picky: if (embeddedPickyMuon_) alreadyEmbedded = true; break;
        case DYT: if (embeddedDytMuon_) alreadyEmbedded = true; break;
      }
  }
  if (force || !alreadyEmbedded) {
      muonBestTrack_.push_back(*reco::Muon::muonBestTrack());
      embeddedMuonBestTrack_ = true;
  }
}

/// embed the Track selected to be the best measurement of the muon parameters
void Muon::embedTunePMuonBestTrack(bool force) {
  tunePMuonBestTrack_.clear();
  bool alreadyEmbedded = false;
  embeddedTunePMuonBestTrack_ = false;
  if (!force) {
      switch (tunePMuonBestTrackType()) {
          case None: alreadyEmbedded = true; break;
          case InnerTrack: if (embeddedTrack_) alreadyEmbedded = true; break;
          case OuterTrack: if (embeddedStandAloneMuon_) alreadyEmbedded = true; break;
          case CombinedTrack: if (embeddedCombinedMuon_) alreadyEmbedded = true; break;
          case TPFMS: if (embeddedTpfmsMuon_) alreadyEmbedded = true; break;
          case Picky: if (embeddedPickyMuon_) alreadyEmbedded = true; break;
          case DYT: if (embeddedDytMuon_) alreadyEmbedded = true; break;
      }
      if (muonBestTrackType() == tunePMuonBestTrackType()) {
          if (embeddedMuonBestTrack_) alreadyEmbedded = true;
      }
  }
  if (force || !alreadyEmbedded) {
      tunePMuonBestTrack_.push_back(*reco::Muon::tunePMuonBestTrack());
      embeddedTunePMuonBestTrack_ = true;
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

/// embed the MuonMETCorrectionData for muon corrected caloMET
void Muon::embedCaloMETMuonCorrs(const reco::MuonMETCorrectionData& t) {
  caloMETMuonCorrs_.clear();
  caloMETMuonCorrs_.push_back(t);
  embeddedCaloMETMuonCorrs_ = true;
}

/// embed the MuonMETCorrectionData for tcMET
void Muon::embedTcMETMuonCorrs(const reco::MuonMETCorrectionData& t) {
  tcMETMuonCorrs_.clear();
  tcMETMuonCorrs_.push_back(t);
  embeddedTCMETMuonCorrs_ = true;
}

/// embed the picky Track
void Muon::embedPickyMuon() {
  pickyMuon_.clear();
  reco::TrackRef tk = reco::Muon::pickyTrack();
  if (tk.isNonnull()) {
    pickyMuon_.push_back(*tk);
    embeddedPickyMuon_ = true;
  }
}

/// embed the tpfms Track
void Muon::embedTpfmsMuon() {
  tpfmsMuon_.clear();
  reco::TrackRef tk = reco::Muon::tpfmsTrack();
  if (tk.isNonnull()) {
    tpfmsMuon_.push_back(*tk);
    embeddedTpfmsMuon_ = true;
  }
}

/// embed the dyt Track
void Muon::embedDytMuon() {
  dytMuon_.clear();
  reco::TrackRef tk = reco::Muon::dytTrack();
  if (tk.isNonnull()) {
    dytMuon_.push_back(*tk);
    embeddedDytMuon_ = true;
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
    reco::TrackRef t = globalTrack();
    return t->chi2() / t->ndof();
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
    reco::TrackRef t = innerTrack();
    return t->numberOfValidHits();
  }
}

// embed various impact parameters with errors
// IpType defines the type of the impact parameter
double Muon::dB(IpType type_) const {
  // more IP types (new)
  if ( cachedIP_ & (1 << int(type_))) {
    return ip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }
}


// embed various impact parameters with errors
// IpType defines the type of the impact parameter
double Muon::edB(IpType type_) const {
  // more IP types (new)
  if ( cachedIP_ & (1 << int(type_))) {
    return eip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }
}


double Muon::segmentCompatibility(reco::Muon::ArbitrationType arbitrationType) const {
   return muon::segmentCompatibility(*this, arbitrationType);
}

// Selectors
bool Muon::isTightMuon(const reco::Vertex&vtx) const {
  return muon::isTightMuon(*this, vtx);
}

bool Muon::isLooseMuon() const {
  return muon::isLooseMuon(*this);

}

bool Muon::isMediumMuon() const {
  return muon::isMediumMuon(*this);

}

bool Muon::isSoftMuon(const reco::Vertex& vtx) const {
  return muon::isSoftMuon(*this, vtx);
}


bool Muon::isHighPtMuon(const reco::Vertex& vtx) const{
  return muon::isHighPtMuon(*this, vtx);
}

