#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace reco::tau {

namespace {
  const reco::Track* getTrack(const Candidate& cand) 
  { 
    const PFCandidate* pfCandPtr = dynamic_cast<const PFCandidate*>(&cand);
    if (pfCandPtr) {
      // Get the KF track if it exists.  Otherwise, see if PFCandidate has a GSF track.
      if      ( pfCandPtr->trackRef().isNonnull()    ) return pfCandPtr->trackRef().get();
      else if ( pfCandPtr->gsfTrackRef().isNonnull() ) return pfCandPtr->gsfTrackRef().get();
      else return nullptr;
    }

    const pat::PackedCandidate* packedCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if (packedCand && packedCand->hasTrackDetails())
    	return &packedCand->pseudoTrack();

    return nullptr;
  }

  const reco::TrackRef getTrackRef(const Candidate& cand) 
  {
    const PFCandidate* pfCandPtr = dynamic_cast<const PFCandidate*>(&cand);
    if (pfCandPtr)
      return pfCandPtr->trackRef();

    return reco::TrackRef();
  }

  const reco::TrackBaseRef getGsfTrackRef(const Candidate& cand) 
  {
    const PFCandidate* pfCandPtr = dynamic_cast<const PFCandidate*>(&cand);
    if (pfCandPtr) {
      return reco::TrackBaseRef(pfCandPtr->gsfTrackRef());
    }
    return reco::TrackBaseRef();

  }


  // Translate GsfTrackRef to TrackBaseRef
  template <typename T>
  reco::TrackBaseRef convertRef(const T& ref) {
    return reco::TrackBaseRef(ref);
  }
}

// Quality cut implementations
namespace qcuts {

bool ptMin(const TrackBaseRef& track, double cut) 
{
  LogDebug("TauQCuts") << "<ptMin>: Pt = " << track->pt() << ", cut = " << cut ;
  return (track->pt() > cut);
}

bool ptMin_cand(const Candidate& cand, double cut) 
{
  LogDebug("TauQCuts") << "<ptMin_cand>: Pt = " << cand.pt() << ", cut = " << cut ;
  return (cand.pt() > cut);
}

bool etMin_cand(const Candidate& cand, double cut) 
{
  LogDebug("TauQCuts") << "<etMin_cand>: Et = " << cand.et() << ", cut = " << cut ;
  return (cand.et() > cut);
}

bool trkPixelHits(const Track* track, int cut) 
{
  // For some reason, the number of hits is signed
  LogDebug("TauQCuts") << "<trkPixelHits>: #Pxl hits = " << track->hitPattern().numberOfValidPixelHits() << ", cut = " << cut ;
  return (track->hitPattern().numberOfValidPixelHits() >= cut);
}

bool trkPixelHits_cand(const Candidate& cand, int cut) 
{
  // For some reason, the number of hits is signed
  auto track = getTrack(cand);
  if ( track ) {
    LogDebug("TauQCuts") << "<trkPixelHits_cand>: #Pxl hits = " << trkPixelHits(track, cut) << ", cut = " << cut ;
    return trkPixelHits(track, cut);
  } else {
    LogDebug("TauQCuts") << "<trkPixelHits_cand>: #Pxl hits = N/A, cut = " << cut ;
    return false;
  }
}

bool trkTrackerHits(const Track* track, int cut) 
{
  LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut ;
  return (track->hitPattern().numberOfValidHits() >= cut);
}

bool trkTrackerHits_cand(const Candidate& cand, int cut) 
{
  auto track = getTrack(cand);
  if ( track ) {
    LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut ;
    return trkTrackerHits(track, cut);
  } else {
    LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = N/A, cut = " << cut ;
    return false;
  }
}

bool trkTransverseImpactParameter(const Track* track, const reco::VertexRef* pv, double cut) 
{
  if ( pv->isNull() ) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - trkTransverseImpactParameter";
    return false;
  }
  LogDebug("TauQCuts") << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() ;
  LogDebug("TauQCuts") << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() ;
  LogDebug("TauQCuts") << "--> dxy = " << std::fabs(track->dxy((*pv)->position())) << " (cut = " << cut << ")" ;
  return (std::fabs(track->dxy((*pv)->position())) <= cut);
}

bool trkTransverseImpactParameter_cand(const Candidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrack(cand);
  if ( track ) {
    return trkTransverseImpactParameter(track, pv, cut);
  } else {
    LogDebug("TauQCuts") << "<trkTransverseImpactParameter_cand>: dXY = N/A, cut = " << cut ;
    return false;
  }
}

bool trkLongitudinalImpactParameter(const TrackBase* track, const reco::VertexRef* pv, double cut) 
{
  if ( pv->isNull() ) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - trkLongitudinalImpactParameter";
    return false;
  }
  LogDebug("TauQCuts") << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() ;
  LogDebug("TauQCuts") << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() ;
  LogDebug("TauQCuts") << "--> dz = " << std::fabs(track->dz((*pv)->position())) << " (cut = " << cut << ")" ;
  return (std::fabs(track->dz((*pv)->position())) <= cut);
}

bool trkLongitudinalImpactParameter_cand(const Candidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrack(cand);
  if ( track ) {
    return trkLongitudinalImpactParameter(track, pv, cut);
  } else {
    LogDebug("TauQCuts") << "<trkLongitudinalImpactParameter_cand>: dZ = N/A, cut = " << cut ;
    return false;
  }
}

/// DZ cut, with respect to the current lead rack
bool trkLongitudinalImpactParameterWrtTrack(const Track* track, const Track* leadTrack, const reco::VertexRef* pv, double cut)
{
  if (!leadTrack) {
    edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
      "RecoTauQualityCuts is invalid. - trkLongitudinalImpactParameterWrtTrack";
    return false;
  }
  return (std::fabs(track->dz((*pv)->position()) - leadTrack->dz((*pv)->position())) <= cut);
}

bool trkLongitudinalImpactParameterWrtTrack_cand(const Candidate& cand, const reco::Track* leadTrack, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrack(cand);
  if ( track ) return trkLongitudinalImpactParameterWrtTrack(track, leadTrack, pv, cut);
  else return false;
}

bool minTrackVertexWeight(const TrackBaseRef& track, const reco::VertexRef* pv, double cut) 
{
  if ( pv->isNull() ) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - minTrackVertexWeight";
    return false;
  }
  LogDebug("TauQCuts") << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() ;
  LogDebug("TauQCuts") << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() ;
  LogDebug("TauQCuts") << "--> trackWeight = " << (*pv)->trackWeight(track) << " (cut = " << cut << ")" ;
  return ((*pv)->trackWeight(track) >= cut);
}

bool minTrackVertexWeight(const TrackRef& track, const reco::VertexRef* pv, double cut) 
{
  if ( pv->isNull() ) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - minTrackVertexWeight";
    return false;
  }
  LogDebug("TauQCuts") << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() ;
  LogDebug("TauQCuts") << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() ;
  LogDebug("TauQCuts") << "--> trackWeight = " << (*pv)->trackWeight(track) << " (cut = " << cut << ")" ;
  return ((*pv)->trackWeight(track) >= cut);
}

bool minPackedCandVertexWeight(const pat::PackedCandidate& pCand, const reco::VertexRef* pv, double cut) {

  if ( pv->isNull() ) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - minPackedCandVertexWeight";
    return false;
  }
  //there is some low granular information on track weight in the vertex available with packed cands
  double weight = -9.9;
  if( pCand.vertexRef().isNonnull() && pCand.vertexRef().key() == pv->key() ){
    int quality = pCand.pvAssociationQuality();
    if( quality == pat::PackedCandidate::UsedInFitTight ) weight = 0.6;//0.6 as proxy for weight above 0.5
    else if( quality == pat::PackedCandidate::UsedInFitLoose ) weight = 0.1;//0.6 as proxy for weight below 0.5
  }
  LogDebug("TauQCuts") << " packedCand: Pt = " << pCand.pt() << ", eta = " << pCand.eta() << ", phi = " << pCand.phi() ;
  LogDebug("TauQCuts") << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() ;
  LogDebug("TauQCuts") << "--> trackWeight from packedCand = " << weight << " (cut = " << cut << ")" ;
  return (weight >= cut);
}

bool minTrackVertexWeight_cand(const Candidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return minTrackVertexWeight(track, pv, cut);
  }
  auto gsfTrack = getGsfTrackRef(cand);
  if ( gsfTrack.isNonnull() ) {
    return minTrackVertexWeight(gsfTrack, pv, cut);
  }

  const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
  if( pCand != nullptr && cand.charge() != 0) {
    return minPackedCandVertexWeight(*pCand, pv, cut);
  }
  LogDebug("TauQCuts") << "<minTrackVertexWeight_cand>: weight = N/A, cut = " << cut ;
  return false;
}

bool trkChi2(const Track* track, double cut) 
{
  LogDebug("TauQCuts") << "<trkChi2>: chi^2 = " << track->normalizedChi2() << ", cut = " << cut ;
  return (track->normalizedChi2() <= cut);
}

bool trkChi2_cand(const Candidate& cand, double cut) 
{
  auto track = getTrack(cand);
  if ( track ) {
    LogDebug("TauQCuts") << "<trkChi2_cand>: chi^2 = " << track->normalizedChi2() << ", cut = " << cut ;
    return trkChi2(track, cut);
  } else {
    LogDebug("TauQCuts") << "<trkChi2_cand>: chi^2 = N/A, cut = " << cut ;
    return false;
  }
}

// And a set of qcuts
bool AND(const TrackBaseRef& track, const RecoTauQualityCuts::TrackQCutFuncCollection& cuts) 
{
  for(auto const& func : cuts ) {
    if ( !func(track) ) return false;
  }
  return true;
}

bool AND_cand(const Candidate& cand, const RecoTauQualityCuts::CandQCutFuncCollection& cuts) 
{
  for(auto const& func : cuts ) {
    if ( !func(cand) ) return false;
  }
  return true;
}

// Get the set of Q cuts for a given type (i.e. gamma)
bool mapAndCutByType(const Candidate& cand, const RecoTauQualityCuts::CandQCutFuncMap& funcMap) 
{
  // Find the cuts that for this particle type
  RecoTauQualityCuts::CandQCutFuncMap::const_iterator cuts = funcMap.find(std::abs(cand.pdgId()));
  // Return false if we dont' know how to deal with this particle type
  if ( cuts == funcMap.end() ) return false; 
  return AND_cand(cand, cuts->second); // Otherwise AND all the cuts
}

} // end qcuts implementation namespace

RecoTauQualityCuts::RecoTauQualityCuts(const edm::ParameterSet &qcuts) 
{
  // Setup all of our predicates
  CandQCutFuncCollection chargedHadronCuts;
  CandQCutFuncCollection gammaCuts;
  CandQCutFuncCollection neutralHadronCuts;

  // Make sure there are no extra passed options
  std::set<std::string> passedOptionSet;
  std::vector<std::string> passedOptions = qcuts.getParameterNames();

  for(auto const& option : passedOptions) {
    passedOptionSet.insert(option);
  }

  unsigned int nCuts = 0;
  auto getDouble = [&qcuts, &passedOptionSet, &nCuts](const std::string& name) {
    if(qcuts.exists(name)) {
      ++nCuts;
      passedOptionSet.erase(name);
      return qcuts.getParameter<double>(name);
    }
    return -1.0;
  };
  auto getUint = [&qcuts, &passedOptionSet, &nCuts](const std::string& name) -> unsigned int {
    if(qcuts.exists(name)) {
      ++nCuts;
      passedOptionSet.erase(name);
      return qcuts.getParameter<unsigned int>(name);
    }
    return 0;
  };

  // Build all the QCuts for tracks
  minTrackPt_ = getDouble("minTrackPt");
  maxTrackChi2_ = getDouble("maxTrackChi2");
  minTrackPixelHits_ = getUint("minTrackPixelHits");
  minTrackHits_ = getUint("minTrackHits");
  maxTransverseImpactParameter_ = getDouble("maxTransverseImpactParameter");
  maxDeltaZ_ = getDouble("maxDeltaZ");
  maxDeltaZToLeadTrack_ = getDouble("maxDeltaZToLeadTrack");
  // Require tracks to contribute a minimum weight to the associated vertex.
  minTrackVertexWeight_ = getDouble("minTrackVertexWeight");
  
  // Use bit-wise & to avoid conditional code
  checkHitPattern_ = (minTrackHits_ > 0) || (minTrackPixelHits_ > 0);
  checkPV_ = (maxTransverseImpactParameter_ >= 0) ||
             (maxDeltaZ_ >= 0) ||
             (maxDeltaZToLeadTrack_ >= 0) ||
             (minTrackVertexWeight_ >= 0);

  // Build the QCuts for gammas
  minGammaEt_ = getDouble("minGammaEt");

  // Build QCuts for netural hadrons
  minNeutralHadronEt_ = getDouble("minNeutralHadronEt");

  // Check if there are any remaining unparsed QCuts
  if ( !passedOptionSet.empty() ) {
    std::string unParsedOptions;
    bool thereIsABadParameter = false;
    for(auto const& option : passedOptionSet ) {
      // Workaround for HLT - TODO FIXME
      if ( option == "useTracksInsteadOfPFHadrons" ) {
        // Crash if true - no one should have this option enabled.
        if ( qcuts.getParameter<bool>("useTracksInsteadOfPFHadrons") ) {
          throw cms::Exception("DontUseTracksInQcuts")
            << "The obsolete exception useTracksInsteadOfPFHadrons "
            << "is set to true in the quality cut config." << std::endl;
        }
        continue;
      }
      
      // If we get to this point, there is a real unknown parameter
      thereIsABadParameter = true;
      
      unParsedOptions += option;
      unParsedOptions += "\n";
    }
    if ( thereIsABadParameter ) {
      throw cms::Exception("BadQualityCutConfig")
        << " The PSet passed to the RecoTauQualityCuts class had"
        << " the following unrecognized options: " << std::endl
        << unParsedOptions;
    }
  }

  // Make sure there are at least some quality cuts
  if ( !nCuts ) {
    throw cms::Exception("BadQualityCutConfig")
      << " No options were passed to the quality cut class!" << std::endl;
  }
}

std::pair<edm::ParameterSet, edm::ParameterSet> factorizePUQCuts(const edm::ParameterSet& input) 
{
  edm::ParameterSet puCuts;
  edm::ParameterSet nonPUCuts;

  std::vector<std::string> inputNames = input.getParameterNames();
  for(auto const& cut : inputNames ) {
    if ( cut == "minTrackVertexWeight" || 
	 cut == "maxDeltaZ"            ||
	 cut == "maxDeltaZToLeadTrack" ) {
      puCuts.copyFrom(input, cut);
    } else {
      nonPUCuts.copyFrom(input, cut);
    }
  }
  return std::make_pair(puCuts, nonPUCuts);
}

bool RecoTauQualityCuts::filterTrack(const reco::TrackBaseRef& track) const
{
  if (!filterTrack_(track.get()))
    return false;
  if(minTrackVertexWeight_ >= 0. && !(pv_->trackWeight(convertRef(track)) >= minTrackVertexWeight_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterTrack(const reco::TrackRef& track) const
{
  if (!filterTrack_(track.get()))
    return false;
  if(minTrackVertexWeight_ >= 0. && !(pv_->trackWeight(convertRef(track)) >= minTrackVertexWeight_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterTrack(const reco::Track& track) const
{
  return filterTrack_(&track);
}

bool RecoTauQualityCuts::filterTrack_(const reco::Track* track) const
{
  if(minTrackPt_ >= 0 && !(track->pt() > minTrackPt_)) return false;
  if(maxTrackChi2_ >= 0 && !(track->normalizedChi2() <= maxTrackChi2_)) return false;
  if(checkHitPattern_) {
    const reco::HitPattern &hitPattern = track->hitPattern();
    if(minTrackPixelHits_ > 0 && !(hitPattern.numberOfValidPixelHits() >= minTrackPixelHits_)) return false;
    if(minTrackHits_ > 0 && !(hitPattern.numberOfValidHits() >= minTrackHits_)) return false;
  }
  if(checkPV_ && pv_.isNull()) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
      "RecoTauQualityCuts is invalid. - filterTrack";
    return false;
  }

  if(maxTransverseImpactParameter_ >= 0 &&
     !(std::fabs(track->dxy(pv_->position())) <= maxTransverseImpactParameter_))
      return false;
  if(maxDeltaZ_ >= 0 && !(std::fabs(track->dz(pv_->position())) <= maxDeltaZ_)) return false;
  if(maxDeltaZToLeadTrack_ >= 0) {
    if ( !leadTrack_) {
      edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
        "RecoTauQualityCuts is invalid. - filterTrack";
      return false;
    }

    if(!(std::fabs(track->dz(pv_->position()) - leadTrack_->dz(pv_->position())) <= maxDeltaZToLeadTrack_))
      return false;
  }

  return true;
}

bool RecoTauQualityCuts::filterChargedCand(const reco::Candidate& cand) const {

  if (cand.charge() == 0)
    return true;
  const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
  if (pCand == nullptr)
    return true;

  //Get track, it should be present for cands with pT(charged)>0.5GeV
  //and check track quality critera other than vertex weight
  auto track = getTrack(cand);
  if (track != nullptr){
    if (!filterTrack(*track))
      return false;
  } else {//Candidates without track (pT(charged)<0.5GeV): Can still check pT and calculate dxy and dz
    if(minTrackPt_ >= 0 && !(pCand->pt() > minTrackPt_)) return false;
    if(checkPV_ && pv_.isNull()) {
      edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
	"RecoTauQualityCuts is invalid. - filterChargedCand";
      return false;
    }

    if(maxTransverseImpactParameter_ >= 0 &&
       !(std::fabs(pCand->dxy(pv_->position())) <= maxTransverseImpactParameter_))
      return false;
    if(maxDeltaZ_ >= 0 && !(std::fabs(pCand->dz(pv_->position())) <= maxDeltaZ_)) return false;
    if(maxDeltaZToLeadTrack_ >= 0) {
      if ( leadTrack_ == nullptr) {
	edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
	  "RecoTauQualityCuts is invalid. - filterChargedCand";
	return false;
      }

      if(!(std::fabs(pCand->dz(pv_->position()) - leadTrack_->dz(pv_->position())) <= maxDeltaZToLeadTrack_))
	return false;
    }
  }
  if(minTrackVertexWeight_ >= 0. &&
     !(qcuts::minPackedCandVertexWeight(*pCand, &pv_,  minTrackVertexWeight_)))
     return false;

  return true;
}

bool RecoTauQualityCuts::filterGammaCand(const reco::Candidate& cand) const {
  if(minGammaEt_ >= 0 && !(cand.et() > minGammaEt_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterNeutralHadronCand(const reco::Candidate& cand) const {
  if(minNeutralHadronEt_ >= 0 && !(cand.et() > minNeutralHadronEt_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterCandByType(const reco::Candidate& cand) const {
  switch(std::abs(cand.pdgId())) {
  case 22:
    return filterGammaCand(cand);
  case 130:
    return filterNeutralHadronCand(cand);
  // We use the same qcuts for muons/electrons and charged hadrons.
  case 211:
  case 11:
  case 13:
    // no cuts ATM (track cuts applied in filterCand)
    return true;
  // Return false if we dont' know how to deal with this particle type
  default:
    return false;
  };
  return false;
}

bool RecoTauQualityCuts::filterCand(const reco::Candidate& cand) const 
{
  auto trackRef = getTrackRef(cand);
  bool result = true;

  if (trackRef.isNonnull()) {
    result = filterTrack(trackRef);
  } 
  else {
    auto gsfTrackRef = getGsfTrackRef(cand);
    if (gsfTrackRef.isNonnull())
      result = filterTrack(gsfTrackRef);
    else if (cand.charge() != 0) {
      result = filterChargedCand(cand);
    }
  }
  
  if(result)
    result = filterCandByType(cand);

  return result;
}

void RecoTauQualityCuts::setLeadTrack(const reco::Track& leadTrack) const 
{
  leadTrack_ = &leadTrack;
}

void RecoTauQualityCuts::setLeadTrack(const reco::Candidate& leadCand) const 
{
  leadTrack_ = getTrack(leadCand);
}

void RecoTauQualityCuts::setLeadTrack(const reco::CandidateRef& leadCand) const 
{
  if ( leadCand.isNonnull() ) {
    leadTrack_ = getTrack(*leadCand);
  } else {
    // Set null
    leadTrack_ = nullptr;
  }
}

} // end namespace reco::tau
