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
}

// Quality cut implementations
namespace qcuts {

bool ptMin(const TrackBaseRef& track, double cut) 
{
  //std::cout << "<ptMin>: Pt = " << track->pt() << ", cut = " << cut << std::endl;
  return (track->pt() > cut);
}

bool ptMin_cand(const PFCandidate& cand, double cut) 
{
  //std::cout << "<ptMin_cand>: Pt = " << cand.pt() << ", cut = " << cut << std::endl;
  return (cand.pt() > cut);
}

bool etMin_cand(const PFCandidate& cand, double cut) 
{
  //std::cout << "<etMin_cand>: Et = " << cand.et() << ", cut = " << cut << std::endl;
  return (cand.et() > cut);
}

bool trkPixelHits(const TrackBaseRef& track, int cut) 
{
  // For some reason, the number of hits is signed
  //std::cout << "<trkPixelHits>: #Pxl hits = " << track->hitPattern().numberOfValidPixelHits() << ", cut = " << cut << std::endl;
  return (track->hitPattern().numberOfValidPixelHits() >= cut);
}

bool trkPixelHits_cand(const PFCandidate& cand, int cut) 
{
  // For some reason, the number of hits is signed
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    //std::cout << "<trkPixelHits_cand>: #Pxl hits = " << trkPixelHits(track, cut) << ", cut = " << cut << std::endl;
    return trkPixelHits(track, cut);
  } else {
    //std::cout << "<trkPixelHits_cand>: #Pxl hits = N/A, cut = " << cut << std::endl;
    return false;
  }
}

bool trkTrackerHits(const TrackBaseRef& track, int cut) 
{
  //std::cout << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut << std::endl;
  return (track->hitPattern().numberOfValidHits() >= cut);
}

bool trkTrackerHits_cand(const PFCandidate& cand, int cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    //std::cout << "<trkTrackerHits>: #Trk hits = " << track->hitPattern().numberOfValidHits() << ", cut = " << cut << std::endl;
    return trkTrackerHits(track, cut);
  } else {
    //std::cout << "<trkTrackerHits>: #Trk hits = N/A, cut = " << cut << std::endl;
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
  //std::cout << "<trkTransverseImpactParameter>:" << std::endl;
  //std::cout << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << std::endl;
  //std::cout << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() << std::endl;
  //std::cout << "--> dxy = " << std::fabs(track->dxy((*pv)->position())) << " (cut = " << cut << ")" << std::endl;
  return (std::fabs(track->dxy((*pv)->position())) <= cut);
}

bool trkTransverseImpactParameter_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return trkTransverseImpactParameter(track, pv, cut);
  } else {
    //std::cout << "<trkTransverseImpactParameter_cand>: dXY = N/A, cut = " << cut << std::endl;
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
  //std::cout << "<trkLongitudinalImpactParameter>:" << std::endl;
  //std::cout << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << std::endl;
  //std::cout << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() << std::endl;
  //std::cout << "--> dz = " << std::fabs(track->dz((*pv)->position())) << " (cut = " << cut << ")" << std::endl;
  return (std::fabs(track->dz((*pv)->position())) <= cut);
}

bool trkLongitudinalImpactParameter_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return trkLongitudinalImpactParameter(track, pv, cut);
  } else {
    //std::cout << "<trkLongitudinalImpactParameter_cand>: dZ = N/A, cut = " << cut << std::endl;
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
  //std::cout << "<minTrackVertexWeight>:" << std::endl;
  //std::cout << " track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << std::endl;
  //std::cout << " vertex: x = " << (*pv)->position().x() << ", y = " << (*pv)->position().y() << ", z = " << (*pv)->position().z() << std::endl;
  //std::cout << "--> trackWeight = " << (*pv)->trackWeight(track) << " (cut = " << cut << ")" << std::endl;
  return ((*pv)->trackWeight(track) >= cut);
}

bool minTrackVertexWeight_cand(const PFCandidate& cand, const reco::VertexRef* pv, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    return minTrackVertexWeight(track, pv, cut);
  } else {
    //std::cout << "<minTrackVertexWeight_cand>: weight = N/A, cut = " << cut << std::endl;
    return false;
  }
}

bool trkChi2(const TrackBaseRef& track, double cut) 
{
  //std::cout << "<trkChi2>: chi^2 = " << track->normalizedChi2() << ", cut = " << cut << std::endl;
  return (track->normalizedChi2() <= cut);
}

bool trkChi2_cand(const PFCandidate& cand, double cut) 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) {
    //std::cout << "<trkChi2_cand>: chi^2 = " << track->normalizedChi2() << ", cut = " << cut << std::endl;
    return trkChi2(track, cut);
  } else {
    //std::cout << "<trkChi2_cand>: chi^2 = N/A, cut = " << cut << std::endl;
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

  // Build all the QCuts for tracks
  if ( qcuts.exists("minTrackPt") ) {
    trackQCuts_.push_back(boost::bind(qcuts::ptMin, _1, qcuts.getParameter<double>("minTrackPt")));
    passedOptionSet.erase("minTrackPt");
  }

  if ( qcuts.exists("maxTrackChi2") ) {
    trackQCuts_.push_back(boost::bind(qcuts::trkChi2, _1, qcuts.getParameter<double>("maxTrackChi2")));
    passedOptionSet.erase("maxTrackChi2");
  }

  if ( qcuts.exists("minTrackPixelHits") ) {
    uint32_t minTrackPixelHits = qcuts.getParameter<uint32_t>("minTrackPixelHits");
    if ( minTrackPixelHits >= 1 ) {
      trackQCuts_.push_back(boost::bind(qcuts::trkPixelHits, _1, minTrackPixelHits));
    }
    passedOptionSet.erase("minTrackPixelHits");
  }

  if ( qcuts.exists("minTrackHits") ) {
    uint32_t minTrackHits = qcuts.getParameter<uint32_t>("minTrackHits");
    if ( minTrackHits >= 1 ) {
      trackQCuts_.push_back(boost::bind(qcuts::trkTrackerHits, _1, minTrackHits));
    }
    passedOptionSet.erase("minTrackHits");
  }

  // The impact parameter functions are bound to our member PV, since they
  // need it to compute the discriminant value.
  if ( qcuts.exists("maxTransverseImpactParameter") ) {
    trackQCuts_.push_back(boost::bind(qcuts::trkTransverseImpactParameter, _1, &pv_, qcuts.getParameter<double>("maxTransverseImpactParameter")));
    passedOptionSet.erase("maxTransverseImpactParameter");
  }

  if ( qcuts.exists("maxDeltaZ") ) {
    trackQCuts_.push_back(boost::bind(qcuts::trkLongitudinalImpactParameter, _1, &pv_, qcuts.getParameter<double>("maxDeltaZ")));
    passedOptionSet.erase("maxDeltaZ");
  }

  if ( qcuts.exists("maxDeltaZToLeadTrack") ) {
    trackQCuts_.push_back(boost::bind(qcuts::trkLongitudinalImpactParameterWrtTrack, _1, &leadTrack_, &pv_, qcuts.getParameter<double>("maxDeltaZToLeadTrack")));
    passedOptionSet.erase("maxDeltaZToLeadTrack");
  }

  // Require tracks to contribute a minimum weight to the associated vertex.
  if ( qcuts.exists("minTrackVertexWeight") ) {
    double minTrackVertexWeight = qcuts.getParameter<double>("minTrackVertexWeight");
    if ( minTrackVertexWeight > -1. ) {
      trackQCuts_.push_back(boost::bind(qcuts::minTrackVertexWeight, _1, &pv_, minTrackVertexWeight));
    }
    passedOptionSet.erase("minTrackVertexWeight");
  }

  // Build the QCuts for gammas
  if ( qcuts.exists("minGammaEt") ) {
    gammaCuts.push_back(boost::bind(qcuts::etMin_cand, _1, qcuts.getParameter<double>("minGammaEt")));
    passedOptionSet.erase("minGammaEt");
  }

  // Build QCuts for netural hadrons
  if ( qcuts.exists("minNeutralHadronEt") ) {
    neutralHadronCuts.push_back(boost::bind(qcuts::etMin_cand, _1, qcuts.getParameter<double>("minNeutralHadronEt")));
    passedOptionSet.erase("minNeutralHadronEt");
  }

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
  size_t nCuts = chargedHadronCuts.size() + gammaCuts.size() + neutralHadronCuts.size() + trackQCuts_.size();
  if ( !nCuts ) {
    throw cms::Exception("BadQualityCutConfig")
      << " No options were passed to the quality cut class!" << std::endl;
  }

  // Build final level predicate that works on Tracks
  trackPredicate_ = boost::bind(qcuts::AND, _1, boost::cref(trackQCuts_));

  // Map our QCut collections to the particle Ids they are associated to.
  candQCuts_[PFCandidate::h]     = chargedHadronCuts;
  candQCuts_[PFCandidate::gamma] = gammaCuts;
  candQCuts_[PFCandidate::h0]    = neutralHadronCuts;
  // We use the same qcuts for muons/electrons and charged hadrons.
  candQCuts_[PFCandidate::e]     = chargedHadronCuts;
  candQCuts_[PFCandidate::mu]    = chargedHadronCuts;

  // Build a final level predicate that works on any PFCand
  candPredicate_ = boost::bind(qcuts::mapAndCutByType, _1, boost::cref(candQCuts_));
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
  return trackPredicate_(track);
}

bool RecoTauQualityCuts::filterCand(const reco::PFCandidate& cand) const 
{
  auto track = getTrackRef(cand);
  if ( track.isNonnull() ) return (trackPredicate_(track) & candPredicate_(cand));
  return candPredicate_(cand);
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
