#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaL1TkIsolation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

EgammaL1TkIsolation::TrkCuts::TrkCuts(const edm::ParameterSet& para) {
  minPt = para.getParameter<double>("minPt");
  auto sq = [](double val) { return val * val; };
  minDR2 = sq(para.getParameter<double>("minDR"));
  maxDR2 = sq(para.getParameter<double>("maxDR"));
  minDEta = para.getParameter<double>("minDEta");
  maxDZ = para.getParameter<double>("maxDZ");
}

edm::ParameterSetDescription EgammaL1TkIsolation::TrkCuts::pSetDescript() {
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt", 2.0);
  desc.add<double>("maxDR", 0.3);
  desc.add<double>("minDR", 0.01);
  desc.add<double>("minDEta", 0.003);
  desc.add<double>("maxDZ", 0.7);
  return desc;
}

EgammaL1TkIsolation::EgammaL1TkIsolation(const edm::ParameterSet& para)
    : barrelCuts_(para.getParameter<edm::ParameterSet>("barrelCuts")),
      endcapCuts_(para.getParameter<edm::ParameterSet>("endcapCuts")) {}

edm::ParameterSetDescription EgammaL1TkIsolation::pSetDescript() {
  edm::ParameterSetDescription desc;
  desc.add("barrelCuts", TrkCuts::pSetDescript());
  desc.add("endcapCuts", TrkCuts::pSetDescript());
  return desc;
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

  std::cout << "obj eta " << objEta << " obj phi " << objPhi << " objZ " << objZ << std::endl;

  const TrkCuts& cuts = std::abs(objEta) < 1.5 ? barrelCuts_ : endcapCuts_;

  for (auto& trk : tracks) {
    const float trkPt = trk.momentum().perp();
    if (passTrkSel(trk, trkPt, cuts, objEta, objPhi, objZ)) {
      ptSum += trkPt;
      nrTrks++;
    }
  }
  return {nrTrks, ptSum};
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
