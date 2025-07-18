#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"

using namespace reco;

TrackSelector::TrackSelector(const edm::ParameterSet &params)
    : minPixelHits(params.getParameter<unsigned int>("pixelHitsMin")),
      minTotalHits(params.getParameter<unsigned int>("totalHitsMin")),
      minPt(params.getParameter<double>("ptMin")),
      maxNormChi2(params.getParameter<double>("normChi2Max")),
      maxJetDeltaR(params.getParameter<double>("jetDeltaRMax")),
      maxDistToAxis(params.getParameter<double>("maxDistToAxis")),
      maxDecayLen(params.getParameter<double>("maxDecayLen")),
      sip2dValMin(params.getParameter<double>("sip2dValMin")),
      sip2dValMax(params.getParameter<double>("sip2dValMax")),
      sip2dSigMin(params.getParameter<double>("sip2dSigMin")),
      sip2dSigMax(params.getParameter<double>("sip2dSigMax")),
      sip3dValMin(params.getParameter<double>("sip3dValMin")),
      sip3dValMax(params.getParameter<double>("sip3dValMax")),
      sip3dSigMin(params.getParameter<double>("sip3dSigMin")),
      sip3dSigMax(params.getParameter<double>("sip3dSigMax")),
      useVariableJTA_(params.existsAs<bool>("useVariableJTA") ? params.getParameter<bool>("useVariableJTA") : false) {
  std::string qualityClass = params.getParameter<std::string>("qualityClass");
  if (qualityClass == "any" || qualityClass == "Any" || qualityClass == "ANY" || qualityClass.empty()) {
    selectQuality = false;
    quality = reco::TrackBase::undefQuality;
  } else {
    selectQuality = true;
    quality = reco::TrackBase::qualityByName(qualityClass);
  }
  if (useVariableJTA_) {
    varJTApars = {params.getParameter<double>("a_dR"),
                  params.getParameter<double>("b_dR"),
                  params.getParameter<double>("a_pT"),
                  params.getParameter<double>("b_pT"),
                  params.getParameter<double>("min_pT"),
                  params.getParameter<double>("max_pT"),
                  params.getParameter<double>("min_pT_dRcut"),
                  params.getParameter<double>("max_pT_dRcut"),
                  params.getParameter<double>("max_pT_trackPTcut")};
  }
}

bool TrackSelector::operator()(const Track &track,
                               const btag::TrackIPData &ipData,
                               const Jet &jet,
                               const GlobalPoint &pv) const {
  bool jtaPassed = false;
  if (useVariableJTA_) {
    jtaPassed = TrackIPTagInfo::passVariableJTA(
        varJTApars, jet.pt(), track.pt(), reco::deltaR(jet.momentum(), track.momentum()));
  } else
    jtaPassed = true;

  return track.pt() >= minPt && reco::deltaR2(jet.momentum(), track.momentum()) < maxJetDeltaR * maxJetDeltaR &&
         jtaPassed && trackSelection(track, ipData, jet, pv);
}

bool TrackSelector::operator()(const CandidatePtr &track,
                               const btag::TrackIPData &ipData,
                               const Jet &jet,
                               const GlobalPoint &pv) const {
  bool jtaPassed = false;
  if (useVariableJTA_) {
    jtaPassed = TrackIPTagInfo::passVariableJTA(
        varJTApars, jet.pt(), track->pt(), reco::deltaR(jet.momentum(), track->momentum()));
  } else
    jtaPassed = true;

  return track->pt() >= minPt && reco::deltaR2(jet.momentum(), track->momentum()) < maxJetDeltaR * maxJetDeltaR &&
         jtaPassed && trackSelection(*reco::btag::toTrack(track), ipData, jet, pv);
}

bool TrackSelector::trackSelection(const Track &track,
                                   const btag::TrackIPData &ipData,
                                   const Jet &jet,
                                   const GlobalPoint &pv) const {
  return (!selectQuality || track.quality(quality)) &&
         (minPixelHits <= 0 || track.hitPattern().numberOfValidPixelHits() >= (int)minPixelHits) &&
         (minTotalHits <= 0 || track.hitPattern().numberOfValidHits() >= (int)minTotalHits) &&
         track.normalizedChi2() < maxNormChi2 && std::abs(ipData.distanceToJetAxis.value()) <= maxDistToAxis &&
         (ipData.closestToJetAxis - pv).mag() <= maxDecayLen && ipData.ip2d.value() >= sip2dValMin &&
         ipData.ip2d.value() <= sip2dValMax && ipData.ip2d.significance() >= sip2dSigMin &&
         ipData.ip2d.significance() <= sip2dSigMax && ipData.ip3d.value() >= sip3dValMin &&
         ipData.ip3d.value() <= sip3dValMax && ipData.ip3d.significance() >= sip3dSigMin &&
         ipData.ip3d.significance() <= sip3dSigMax;
}

void TrackSelector::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<unsigned int>("pixelHitsMin", 0);
  desc.add<unsigned int>("totalHitsMin", 0);
  desc.add<double>("ptMin", 0.0);
  desc.add<double>("normChi2Max", 99999.9);
  desc.add<double>("jetDeltaRMax", 0.3);
  desc.add<double>("maxDistToAxis", 0.07);
  desc.add<double>("maxDecayLen", 5);
  desc.add<double>("sip2dValMin", -99999.9);
  desc.add<double>("sip2dValMax", 99999.9);
  desc.add<double>("sip2dSigMin", -99999.9);
  desc.add<double>("sip2dSigMax", 99999.9);
  desc.add<double>("sip3dValMin", -99999.9);
  desc.add<double>("sip3dValMax", 99999.9);
  desc.add<double>("sip3dSigMin", -99999.9);
  desc.add<double>("sip3dSigMax", 99999.9);
  desc.add<bool>("useVariableJTA", false);
  desc.add<std::string>("qualityClass", "");
  desc.add<double>("a_dR", -0.001053);
  desc.add<double>("b_dR", 0.6263);
  desc.add<double>("a_pT", 0.005263);
  desc.add<double>("b_pT", 0.3684);
  desc.add<double>("min_pT", 120);
  desc.add<double>("max_pT", 500);
  desc.add<double>("min_pT_dRcut", 0.5);
  desc.add<double>("max_pT_dRcut", 0.1);
  desc.add<double>("max_pT_trackPTcut", 3);
}
