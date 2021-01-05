#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaL1TkIsolation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>

EgammaL1TkIsolation::EgammaL1TkIsolation(const edm::ParameterSet& para)
    : useAbsEta_(para.getParameter<bool>("useAbsEta")),
      etaBoundaries_(para.getParameter<std::vector<double>>("etaBoundaries")) {
  const auto& trkCutParams = para.getParameter<std::vector<edm::ParameterSet>>("trkCuts");
  for (const auto& params : trkCutParams) {
    trkCuts_.emplace_back(TrkCuts(params));
  }
  if (etaBoundaries_.size() + 1 != trkCuts_.size()) {
    throw cms::Exception("ConfigError") << "EgammaL1TkIsolation: etaBoundaries parameters size ("
                                        << etaBoundaries_.size()
                                        << ") should be one less than the size of trkCuts VPSet (" << trkCuts_.size()
                                        << ")";
  }
  if (!std::is_sorted(etaBoundaries_.begin(), etaBoundaries_.end())) {
    throw cms::Exception("ConfigError")
        << "EgammaL1TkIsolation: etaBoundaries parameter's entries should be in increasing value";
  }
}

void EgammaL1TkIsolation::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add("useAbsEta", true);
  desc.add("etaBoundaries", std::vector{1.5});
  desc.addVPSet("trkCuts", TrkCuts::makePSetDescription(), {edm::ParameterSet(), edm::ParameterSet()});
}

std::pair<int, double> EgammaL1TkIsolation::calIsol(const reco::TrackBase& trk, const L1TrackCollection& tracks) const {
  return calIsol(trk.eta(), trk.phi(), trk.vz(), tracks);
}

std::pair<int, double> EgammaL1TkIsolation::calIsol(const double objEta,
                                                    const double objPhi,
                                                    const double objZ,
                                                    const L1TrackCollection& tracks) const {
  double ptSum = 0.;
  int nrTrks = 0;

  const TrkCuts& cuts = trkCuts_[etaBinNr(objEta)];

  for (const auto& trk : tracks) {
    const float trkPt = trk.momentum().perp();
    if (passTrkSel(trk, trkPt, cuts, objEta, objPhi, objZ)) {
      ptSum += trkPt;
      nrTrks++;
    }
  }
  return {nrTrks, ptSum};
}

EgammaL1TkIsolation::TrkCuts::TrkCuts(const edm::ParameterSet& para) {
  minPt = para.getParameter<double>("minPt");
  auto sq = [](double val) { return val * val; };
  minDR2 = sq(para.getParameter<double>("minDR"));
  maxDR2 = sq(para.getParameter<double>("maxDR"));
  minDEta = para.getParameter<double>("minDEta");
  maxDZ = para.getParameter<double>("maxDZ");
}

edm::ParameterSetDescription EgammaL1TkIsolation::TrkCuts::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt", 2.0);
  desc.add<double>("maxDR", 0.3);
  desc.add<double>("minDR", 0.01);
  desc.add<double>("minDEta", 0.003);
  desc.add<double>("maxDZ", 0.7);
  return desc;
}

//as we have verfied that trkCuts_ size is etaBoundaries_ size +1
//then this is always a valid binnr for trkCuts_
size_t EgammaL1TkIsolation::etaBinNr(double eta) const {
  if (useAbsEta_) {
    eta = std::abs(eta);
  }
  auto res = std::upper_bound(etaBoundaries_.begin(), etaBoundaries_.end(), eta);
  size_t binNr = std::distance(etaBoundaries_.begin(), res);
  return binNr;
}

bool EgammaL1TkIsolation::passTrkSel(const L1Track& trk,
                                     const double trkPt,
                                     const TrkCuts& cuts,
                                     const double objEta,
                                     const double objPhi,
                                     const double objZ) {
  if (trkPt > cuts.minPt && std::abs(objZ - trk.z0()) < cuts.maxDZ) {
    const float trkEta = trk.eta();
    const float dEta = trkEta - objEta;
    const float dR2 = reco::deltaR2(objEta, objPhi, trkEta, trk.phi());
    return dR2 >= cuts.minDR2 && dR2 <= cuts.maxDR2 && std::abs(dEta) >= cuts.minDEta;
  }

  return false;
}
