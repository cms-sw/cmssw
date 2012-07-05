#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <boost/bind.hpp>

namespace reco { namespace tau {

namespace {
// Get the KF track if it exists.  Otherwise, see if it has a GSF track.
const reco::TrackBaseRef getTrack(const PFCandidate& cand) {
  if (cand.trackRef().isNonnull())
    return reco::TrackBaseRef(cand.trackRef());
  else if (cand.gsfTrackRef().isNonnull()) {
    return reco::TrackBaseRef(cand.gsfTrackRef());
  }
  return reco::TrackBaseRef();
}
}

// Quality cut implementations
namespace qcuts {

bool ptMin(const PFCandidate& cand, double cut) {
  return cand.pt() > cut;
}

bool etMin(const PFCandidate& cand, double cut) {
  return cand.et() > cut;
}

bool trkPixelHits(const PFCandidate& cand, int cut) {
  // For some reason, the number of hits is signed
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  return trk->hitPattern().numberOfValidPixelHits() >= cut;
}

bool trkTrackerHits(const PFCandidate& cand, int cut) {
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  return trk->hitPattern().numberOfValidHits() >= cut;
}

bool trkTransverseImpactParameter(const PFCandidate& cand,
                                  const reco::VertexRef* pv,
                                  double cut) {
  if (pv->isNull()) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
        "RecoTauQualityCuts is invalid. - trkTransverseImpactParameter";
    return false;
  }
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  return std::abs(trk->dxy((*pv)->position())) <= cut;
}

bool trkLongitudinalImpactParameter(const PFCandidate& cand,
                                    const reco::VertexRef* pv,
                                    double cut) {
  if (pv->isNull()) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
        "RecoTauQualityCuts is invalid. - trkLongitudinalImpactParameter";
    return false;
  }
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  double difference = std::abs(trk->dz((*pv)->position()));
  //std::cout << "QCUTS LIP: track vz: " << trk->vz() <<
    //" diff: " << difference << " cut: " << cut << std::endl;
  return difference <= cut;
}

/// DZ cut, with respect to the current lead rack
bool trkLongitudinalImpactParameterWrtTrack(const PFCandidate& cand,
    const reco::TrackBaseRef* trk, double cut) {
  if (trk->isNull()) {
    edm::LogError("QCutsNoValidLeadTrack") << "Lead track Ref in " <<
        "RecoTauQualityCuts is invalid. - trkLongitudinalImpactParameterWrtTrack";
    return false;
  }
  TrackBaseRef candTrk = getTrack(cand);
  if (!candTrk) return false;
  double difference = std::abs((*trk)->vz() - candTrk->vz());
  return difference <= cut;
}


bool minTrackVertexWeight(const PFCandidate& cand, const reco::VertexRef* pv,
    double cut) {
  if (pv->isNull()) {
    edm::LogError("QCutsNoPrimaryVertex") << "Primary vertex Ref in " <<
        "RecoTauQualityCuts is invalid. - minTrackVertexWeight";
    return false;
  }
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  double weight = (*pv)->trackWeight(trk);
  return weight >= cut;
}

bool trkChi2(const PFCandidate& cand, double cut) {
  TrackBaseRef trk = getTrack(cand);
  if (!trk) return false;
  return trk->normalizedChi2() <= cut;
}

// And a set of qcuts
bool AND(const PFCandidate& cand,
         const RecoTauQualityCuts::QCutFuncCollection& cuts) {
  BOOST_FOREACH(const RecoTauQualityCuts::QCutFunc& func, cuts) {
    if (!func(cand))
      return false;
  }
  return true;
}

// Get the set of Q cuts for a given type (i.e. gamma)
bool mapAndCutByType(const PFCandidate& cand,
                     const RecoTauQualityCuts::QCutFuncMap& funcMap) {
  // Find the cuts that for this particle type
  RecoTauQualityCuts::QCutFuncMap::const_iterator cuts =
      funcMap.find(cand.particleId());
  // Return false if we dont' know how to deal w/ this particle type
  if (cuts == funcMap.end())
    return false;
  else
    // Otherwise AND all the cuts
    return AND(cand, cuts->second);
}

}  // end qcuts implementation namespace

RecoTauQualityCuts::RecoTauQualityCuts(const edm::ParameterSet &qcuts) {
  // Setup all of our predicates
  QCutFuncCollection chargedHadronCuts;
  QCutFuncCollection gammaCuts;
  QCutFuncCollection neutralHadronCuts;

  // Make sure there are no extra passed options
  std::set<std::string> passedOptionSet;
  std::vector<std::string> passedOptions = qcuts.getParameterNames();

  BOOST_FOREACH(const std::string& option, passedOptions) {
    passedOptionSet.insert(option);
  }

  // Build all the QCuts for tracks
  if (qcuts.exists("minTrackPt")) {
    chargedHadronCuts.push_back(
        boost::bind(qcuts::ptMin, _1,
                    qcuts.getParameter<double>("minTrackPt")));
    passedOptionSet.erase("minTrackPt");
  }

  if (qcuts.exists("maxTrackChi2")) {
    chargedHadronCuts.push_back(
        boost::bind(qcuts::trkChi2, _1,
                    qcuts.getParameter<double>("maxTrackChi2")));
    passedOptionSet.erase("maxTrackChi2");
  }

  if (qcuts.exists("minTrackPixelHits")) {
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkPixelHits, _1,
            qcuts.getParameter<uint32_t>("minTrackPixelHits")));
    passedOptionSet.erase("minTrackPixelHits");
  }

  if (qcuts.exists("minTrackHits")) {
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkTrackerHits, _1,
            qcuts.getParameter<uint32_t>("minTrackHits")));
    passedOptionSet.erase("minTrackHits");
  }

  // The impact parameter functions are bound to our member PV, since they
  // need it to compute the discriminant value.
  if (qcuts.exists("maxTransverseImpactParameter")) {
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkTransverseImpactParameter, _1, &pv_,
            qcuts.getParameter<double>("maxTransverseImpactParameter")));
    passedOptionSet.erase("maxTransverseImpactParameter");
  }

  if (qcuts.exists("maxDeltaZ")) {
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkLongitudinalImpactParameter, _1, &pv_,
            qcuts.getParameter<double>("maxDeltaZ")));
    passedOptionSet.erase("maxDeltaZ");
  }

  if (qcuts.exists("maxDeltaZToLeadTrack")) {
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkLongitudinalImpactParameterWrtTrack, _1, &leadTrack_,
            qcuts.getParameter<double>("maxDeltaZToLeadTrack")));
    passedOptionSet.erase("maxDeltaZToLeadTrack");
  }

  // Require tracks to contribute a minimum weight to the associated vertex.
  if (qcuts.exists("minTrackVertexWeight")) {
    chargedHadronCuts.push_back(boost::bind(
          qcuts::minTrackVertexWeight, _1, &pv_,
          qcuts.getParameter<double>("minTrackVertexWeight")));
    passedOptionSet.erase("minTrackVertexWeight");
  }

  // Build the QCuts for gammas
  if (qcuts.exists("minGammaEt")) {
    gammaCuts.push_back(boost::bind(
            qcuts::etMin, _1, qcuts.getParameter<double>("minGammaEt")));
    passedOptionSet.erase("minGammaEt");
  }

  // Build QCuts for netural hadrons
  if (qcuts.exists("minNeutralHadronEt")) {
    neutralHadronCuts.push_back(boost::bind(
            qcuts::etMin, _1,
            qcuts.getParameter<double>("minNeutralHadronEt")));
    passedOptionSet.erase("minNeutralHadronEt");
  }

  // Check if there are any remaining unparsed QCuts
  if (passedOptionSet.size()) {
    std::string unParsedOptions;
    bool thereIsABadParameter = false;
    BOOST_FOREACH(const std::string& option, passedOptionSet) {
      // Workaround for HLT - TODO FIXME
      if (option == "useTracksInsteadOfPFHadrons") {
        // Crash if true - no one should have this option enabled.
        if (qcuts.getParameter<bool>("useTracksInsteadOfPFHadrons")) {
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
    if (thereIsABadParameter) {
      throw cms::Exception("BadQualityCutConfig")
        << " The PSet passed to the RecoTauQualityCuts class had"
        << " the following unrecognized options: " << std::endl
        << unParsedOptions;
    }
  }

  // Make sure there are at least some quality cuts
  size_t nCuts = chargedHadronCuts.size() + gammaCuts.size()
    + neutralHadronCuts.size();
  if (!nCuts) {
    throw cms::Exception("BadQualityCutConfig")
      << " No options were passed to the quality cut class!" << std::endl;
  }

  // Map our QCut collections to the particle Ids they are associated to.
  qcuts_[PFCandidate::h] = chargedHadronCuts;
  qcuts_[PFCandidate::gamma] = gammaCuts;
  qcuts_[PFCandidate::h0] = neutralHadronCuts;
  // We use the same qcuts for muons/electrons and charged hadrons.
  qcuts_[PFCandidate::e] = chargedHadronCuts;
  qcuts_[PFCandidate::mu] = chargedHadronCuts;

  // Build a final level predicate that works on any PFCand
  predicate_ = boost::bind(qcuts::mapAndCutByType, _1, boost::cref(qcuts_));
}

std::pair<edm::ParameterSet, edm::ParameterSet> factorizePUQCuts(
    const edm::ParameterSet& input) {

  edm::ParameterSet puCuts;
  edm::ParameterSet nonPUCuts;

  std::vector<std::string> inputNames = input.getParameterNames();
  BOOST_FOREACH(const std::string& cut, inputNames) {
    if (cut == "minTrackVertexWeight" || cut == "maxDeltaZ"
        || cut == "maxDeltaZToLeadTrack") {
      puCuts.copyFrom(input, cut);
    } else {
      nonPUCuts.copyFrom(input, cut);
    }
  }
  return std::make_pair(puCuts, nonPUCuts);
}


void RecoTauQualityCuts::setLeadTrack(
    const reco::PFCandidate& leadCand) const {
  leadTrack_ = getTrack(leadCand);
}

void RecoTauQualityCuts::setLeadTrack(
    const reco::PFCandidateRef& leadCand) const {
  if (leadCand.isNonnull()) {
    leadTrack_ = getTrack(*leadCand);
  } else {
    // Set null
    leadTrack_ = reco::TrackBaseRef();
  }
}


}}  // end namespace reco::tau
