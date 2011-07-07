#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <boost/bind.hpp>

namespace reco { namespace tau {

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
  TrackRef trk = cand.trackRef();
  if (!trk) return false;
  return trk->hitPattern().numberOfValidPixelHits() >= cut;
}

bool trkTrackerHits(const PFCandidate& cand, int cut) {
  TrackRef trk = cand.trackRef();
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
  TrackRef trk = cand.trackRef();
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
  TrackRef trk = cand.trackRef();
  if (!trk) return false;
  return std::abs(trk->dz((*pv)->position())) <= cut;
}

bool trkChi2(const PFCandidate& cand, double cut) {
  TrackRef trk = cand.trackRef();
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

  // Build all the QCuts for tracks
  if (qcuts.exists("minTrackPt"))
    chargedHadronCuts.push_back(
        boost::bind(qcuts::ptMin, _1,
                    qcuts.getParameter<double>("minTrackPt")));

  if (qcuts.exists("maxTrackChi2"))
    chargedHadronCuts.push_back(
        boost::bind(qcuts::trkChi2, _1,
                    qcuts.getParameter<double>("maxTrackChi2")));

  if (qcuts.exists("minTrackPixelHits"))
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkPixelHits, _1,
            qcuts.getParameter<uint32_t>("minTrackPixelHits")));

  if (qcuts.exists("minTrackHits"))
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkTrackerHits, _1,
            qcuts.getParameter<uint32_t>("minTrackHits")));

  // The impact parameter functions are bound to our member PV, since they
  // need it to compute the discriminant value.
  if (qcuts.exists("maxTransverseImpactParameter"))
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkTransverseImpactParameter, _1, &pv_,
            qcuts.getParameter<double>("maxTransverseImpactParameter")));

  if (qcuts.exists("maxDeltaZ"))
    chargedHadronCuts.push_back(boost::bind(
            qcuts::trkLongitudinalImpactParameter, _1, &pv_,
            qcuts.getParameter<double>("maxDeltaZ")));

  // Build the QCuts for gammas
  if (qcuts.exists("minGammaEt"))
    gammaCuts.push_back(boost::bind(
            qcuts::etMin, _1, qcuts.getParameter<double>("minGammaEt")));

  // Build QCuts for netural hadrons
  if (qcuts.exists("minNeutralHadronEt"))
    neutralHadronCuts.push_back(boost::bind(
            qcuts::etMin, _1,
            qcuts.getParameter<double>("minNeutralHadronEt")));

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

}}  // end namespace reco::tau
