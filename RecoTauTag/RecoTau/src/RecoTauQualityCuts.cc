#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <boost/bind.hpp>

namespace reco { namespace tau {

namespace {
  // Get the KF track if it exists.  Otherwise, see if PFCandidate has a GSF track.
  const reco::TrackBaseRef getTrackRef(const PFCandidate& cand) 
  {
    if ( cand.trackRef().isNonnull() ) return reco::TrackBaseRef(cand.trackRef());
    else if ( cand.gsfTrackRef().isNonnull() ) return reco::TrackBaseRef(cand.gsfTrackRef());
    else return reco::TrackBaseRef();
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

bool ptMin_cand(const PFCandidate& cand, double cut) 
{
  LogDebug("TauQCuts") << "<ptMin_cand>: Pt = " << cand.pt() << ", cut = " << cut ;
  return (cand.pt() > cut);
}

bool etMin_cand(const PFCandidate& cand, double cut) 
{
  LogDebug("TauQCuts") << "<etMin_cand>: Et = " << cand.et() << ", cut = " << cut ;
  return (cand.et() > cut);
}

bool trkPixelHits(const TrackBaseRef& track, int cut) 
{
  // For some reason, the number of hits is signed
  LogDebug("TauQCuts") << "<trkPixelHits>: #Pxl hits = " << track->hitPattern().numberOfValidPixelHits() << ", cut = " << cut ;
  return (track->hitPattern().numberOfValidPixelHits() >= cut);
}

bool trkPixelHits_cand(const PFCandidate& cand, int cut) 
{
  // For some reason, the number of hits is signed
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    LogDebug("TauQCuts") << "<trkPixelHits_cand>: #Pxl hits = " << trkPixelHits(track, cut) << ", cut = " << cut ;
    return trkPixelHits(track, cut);
  } else {
    LogDebug("TauQCuts") << "<trkPixelHits_cand>: #Pxl hits = N/A, cut = " << cut ;
    return false;
  }
}

bool trkTrackerHits(const TrackBaseRef& track, int cut) 
{
  LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut ;
  return (track->hitPattern().numberOfValidHits() >= cut);
}

bool trkTrackerHits_cand(const PFCandidate& cand, int cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut ;
    return trkTrackerHits(track, cut);
  } else {
    LogDebug("TauQCuts") << "<trkTrackerHits>: #Trk hits = N/A, cut = " << cut ;
    return false;
  }
}

bool trkTransverseImpactParameter(const TrackBaseRef& track, const reco::VertexRef* pv, double cut) 
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

bool trkTransverseImpactParameter_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return trkTransverseImpactParameter(track, pv, cut);
  } else {
    LogDebug("TauQCuts") << "<trkTransverseImpactParameter_cand>: dXY = N/A, cut = " << cut ;
    return false;
  }
}

bool trkLongitudinalImpactParameter(const TrackBaseRef& track, const reco::VertexRef* pv, double cut) 
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

bool trkLongitudinalImpactParameter_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return trkLongitudinalImpactParameter(track, pv, cut);
  } else {
    LogDebug("TauQCuts") << "<trkLongitudinalImpactParameter_cand>: dZ = N/A, cut = " << cut ;
    return false;
  }
}

/// DZ cut, with respect to the current lead rack
bool trkLongitudinalImpactParameterWrtTrack(const TrackBaseRef& track, const reco::TrackBaseRef* leadTrack, const reco::VertexRef* pv, double cut) 
{
  if ( leadTrack->isNull()) {
    edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
      "RecoTauQualityCuts is invalid. - trkLongitudinalImpactParameterWrtTrack";
    return false;
  }
  return (std::fabs(track->dz((*pv)->position()) - (*leadTrack)->dz((*pv)->position())) <= cut);
}

bool trkLongitudinalImpactParameterWrtTrack_cand(const PFCandidate& cand, const reco::TrackBaseRef* leadTrack, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) return trkLongitudinalImpactParameterWrtTrack(track, leadTrack, pv, cut);
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

bool minTrackVertexWeight_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return minTrackVertexWeight(track, pv, cut);
  } else {
    LogDebug("TauQCuts") << "<minTrackVertexWeight_cand>: weight = N/A, cut = " << cut ;
    return false;
  }
}

bool trkChi2(const TrackBaseRef& track, double cut) 
{
  LogDebug("TauQCuts") << "<trkChi2>: chi^2 = " << track->normalizedChi2() << ", cut = " << cut ;
  return (track->normalizedChi2() <= cut);
}

bool trkChi2_cand(const PFCandidate& cand, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
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
  BOOST_FOREACH( const RecoTauQualityCuts::TrackQCutFunc& func, cuts ) {
    if ( !func(track) ) return false;
  }
  return true;
}

bool AND_cand(const PFCandidate& cand, const RecoTauQualityCuts::CandQCutFuncCollection& cuts) 
{
  BOOST_FOREACH( const RecoTauQualityCuts::CandQCutFunc& func, cuts ) {
    if ( !func(cand) ) return false;
  }
  return true;
}

// Get the set of Q cuts for a given type (i.e. gamma)
bool mapAndCutByType(const PFCandidate& cand, const RecoTauQualityCuts::CandQCutFuncMap& funcMap) 
{
  // Find the cuts that for this particle type
  RecoTauQualityCuts::CandQCutFuncMap::const_iterator cuts = funcMap.find(cand.particleId());
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

  BOOST_FOREACH(const std::string& option, passedOptions) {
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
  if ( passedOptionSet.size() ) {
    std::string unParsedOptions;
    bool thereIsABadParameter = false;
    BOOST_FOREACH( const std::string& option, passedOptionSet ) {
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
  BOOST_FOREACH( const std::string& cut, inputNames ) {
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
  return filterTrack_(track);
}

bool RecoTauQualityCuts::filterTrack(const reco::TrackRef& track) const
{
  return filterTrack_(track);
}

template <typename T>
bool RecoTauQualityCuts::filterTrack_(const T& trackRef) const
{
  const Track *track = trackRef.get();
  if(minTrackPt_ >= 0 && !(track->pt() > minTrackPt_)) return false;
  if(maxTrackChi2_ >= 0 && !(track->normalizedChi2() <= maxTrackChi2_)) return false;
  if(checkHitPattern_) {
    const reco::HitPattern hitPattern = track->hitPattern();
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
    if ( leadTrack_.isNull()) {
      edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
        "RecoTauQualityCuts is invalid. - filterTrack";
      return false;
    }

    if(!(std::fabs(track->dz(pv_->position()) - leadTrack_->dz(pv_->position())) <= maxDeltaZToLeadTrack_))
      return false;
  }
  if(minTrackVertexWeight_ > -1.0 && !(pv_->trackWeight(convertRef(trackRef)) >= minTrackVertexWeight_)) return false;

  return true;
}

bool RecoTauQualityCuts::filterGammaCand(const reco::PFCandidate& cand) const {
  if(minGammaEt_ >= 0 && !(cand.et() > minGammaEt_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterNeutralHadronCand(const reco::PFCandidate& cand) const {
  if(minNeutralHadronEt_ >= 0 && !(cand.et() > minNeutralHadronEt_)) return false;
  return true;
}

bool RecoTauQualityCuts::filterCandByType(const reco::PFCandidate& cand) const {
  switch(cand.particleId()) {
  case PFCandidate::gamma:
    return filterGammaCand(cand);
  case PFCandidate::h0:
    return filterNeutralHadronCand(cand);
  // We use the same qcuts for muons/electrons and charged hadrons.
  case PFCandidate::h:
  case PFCandidate::e:
  case PFCandidate::mu:
    // no cuts ATM (track cuts applied in filterCand)
    return true;
  // Return false if we dont' know how to deal with this particle type
  default:
    return false;
  };
  return false;
}

bool RecoTauQualityCuts::filterCand(const reco::PFCandidate& cand) const 
{
  auto trackRef = cand.trackRef();
  bool result = true;
  if(trackRef.isNonnull()) {
    result = filterTrack_(trackRef);
  }
  else {
    auto gsfTrackRef = cand.gsfTrackRef();
    if(gsfTrackRef.isNonnull()) {
      result = filterTrack_(gsfTrackRef);
    }
  }
  if(result)
    result = filterCandByType(cand);
  return result;
}

void RecoTauQualityCuts::setLeadTrack(const reco::TrackRef& leadTrack) const 
{
  leadTrack_ = reco::TrackBaseRef(leadTrack);
}

void RecoTauQualityCuts::setLeadTrack(const reco::PFCandidate& leadCand) const 
{
  leadTrack_ = getTrackRef(leadCand);
}

void RecoTauQualityCuts::setLeadTrack(const reco::PFCandidateRef& leadCand) const 
{
  if ( leadCand.isNonnull() ) {
    leadTrack_ = getTrackRef(*leadCand);
  } else {
    // Set null
    leadTrack_ = reco::TrackBaseRef();
  }
}

}} // end namespace reco::tau
