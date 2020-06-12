#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

EleTkIsolFromCands::TrkCuts::TrkCuts(const edm::ParameterSet& para) {
  minPt = para.getParameter<double>("minPt");
  auto sq = [](double val) { return val * val; };
  minDR2 = sq(para.getParameter<double>("minDR"));
  maxDR2 = sq(para.getParameter<double>("maxDR"));
  minDEta = para.getParameter<double>("minDEta");
  maxDZ = para.getParameter<double>("maxDZ");
  minHits = para.getParameter<int>("minHits");
  minPixelHits = para.getParameter<int>("minPixelHits");
  maxDPtPt = para.getParameter<double>("maxDPtPt");

  auto qualNames = para.getParameter<std::vector<std::string> >("allowedQualities");
  auto algoNames = para.getParameter<std::vector<std::string> >("algosToReject");

  for (auto& qualName : qualNames) {
    allowedQualities.push_back(reco::TrackBase::qualityByName(qualName));
  }
  for (auto& algoName : algoNames) {
    algosToReject.push_back(reco::TrackBase::algoByName(algoName));
  }
  std::sort(algosToReject.begin(), algosToReject.end());
}

edm::ParameterSetDescription EleTkIsolFromCands::TrkCuts::pSetDescript() {
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt", 1.0);
  desc.add<double>("maxDR", 0.3);
  desc.add<double>("minDR", 0.000);
  desc.add<double>("minDEta", 0.005);
  desc.add<double>("maxDZ", 0.1);
  desc.add<double>("maxDPtPt", -1);
  desc.add<int>("minHits", 8);
  desc.add<int>("minPixelHits", 1);
  desc.add<std::vector<std::string> >("allowedQualities");
  desc.add<std::vector<std::string> >("algosToReject");
  return desc;
}

EleTkIsolFromCands::EleTkIsolFromCands(const edm::ParameterSet& para)
    : barrelCuts_(para.getParameter<edm::ParameterSet>("barrelCuts")),
      endcapCuts_(para.getParameter<edm::ParameterSet>("endcapCuts")) {}

edm::ParameterSetDescription EleTkIsolFromCands::pSetDescript() {
  edm::ParameterSetDescription desc;
  desc.add("barrelCuts", TrkCuts::pSetDescript());
  desc.add("endcapCuts", TrkCuts::pSetDescript());
  return desc;
}

std::vector<EleTkIsolFromCands::SimpleTrack> EleTkIsolFromCands::preselectTracksWithCuts(
    reco::TrackCollection const& tracks, TrkCuts const& cuts) {
  std::vector<SimpleTrack> outTracks;
  outTracks.reserve(tracks.size());

  for (auto const& trk : tracks) {
    if (passTrackPreselection(trk, cuts)) {
      outTracks.emplace_back(trk);
    }
  }

  return outTracks;
}

std::vector<EleTkIsolFromCands::SimpleTrack> EleTkIsolFromCands::preselectTracksWithCuts(
    pat::PackedCandidateCollection const& cands, TrkCuts const& cuts, PIDVeto pidVeto) {
  std::vector<SimpleTrack> outTracks;
  outTracks.reserve(cands.size());

  for (auto const& cand : cands) {
    if (cand.hasTrackDetails() && cand.charge() != 0 && passPIDVeto(cand.pdgId(), pidVeto)) {
      const reco::Track& trk = cand.pseudoTrack();
      if (passTrackPreselection(trk, cuts)) {
        outTracks.emplace_back(trk);
      }
    }
  }

  return outTracks;
}

EleTkIsolFromCands::PreselectedTracks EleTkIsolFromCands::preselectTracks(reco::TrackCollection const& tracks) const {
  return {
      .withBarrelCuts = preselectTracksWithCuts(tracks, barrelCuts_),
      .withEndcapCuts = preselectTracksWithCuts(tracks, endcapCuts_),
  };
}
EleTkIsolFromCands::PreselectedTracks EleTkIsolFromCands::preselectTracks(pat::PackedCandidateCollection const& cands,
                                                                          PIDVeto pidVeto) const {
  return {
      .withBarrelCuts = preselectTracksWithCuts(cands, barrelCuts_, pidVeto),
      .withEndcapCuts = preselectTracksWithCuts(cands, endcapCuts_, pidVeto),
  };
}

EleTkIsolFromCands::Output EleTkIsolFromCands::operator()(const reco::TrackBase& eleTrk,
                                                          const PreselectedTracks& tracks) const {
  double ptSum = 0.;
  int nrTrks = 0;

  const double eleEta = eleTrk.eta();
  const double elePhi = eleTrk.phi();
  const double eleVz = eleTrk.vz();

  const bool isBarrelElectron = std::abs(eleEta) < 1.5;

  auto const& preselectedTracks = isBarrelElectron ? tracks.withBarrelCuts : tracks.withEndcapCuts;
  auto const& cuts = isBarrelElectron ? barrelCuts_ : endcapCuts_;

  for (auto& trk : preselectedTracks) {
    if (passMatchingToElectron(trk, cuts, eleEta, elePhi, eleVz)) {
      ptSum += trk.pt;
      nrTrks++;
    }
  }
  return {nrTrks, ptSum};
}

bool EleTkIsolFromCands::passPIDVeto(const int pdgId, const EleTkIsolFromCands::PIDVeto veto) {
  int pidAbs = std::abs(pdgId);
  switch (veto) {
    case PIDVeto::NONE:
      return true;
    case PIDVeto::ELES:
      if (pidAbs == 11)
        return false;
      else
        return true;
    case PIDVeto::NONELES:
      if (pidAbs == 11)
        return true;
      else
        return false;
  }
  throw cms::Exception("CodeError") << "invalid PIDVeto " << static_cast<int>(veto) << ", "
                                    << "this is likely due to some static casting of invalid ints somewhere";
}

EleTkIsolFromCands::PIDVeto EleTkIsolFromCands::pidVetoFromStr(const std::string& vetoStr) {
  if (vetoStr == "NONE")
    return PIDVeto::NONE;
  else if (vetoStr == "ELES")
    return PIDVeto::ELES;
  else if (vetoStr == "NONELES")
    return PIDVeto::NONELES;
  else {
    throw cms::Exception("CodeError") << "unrecognised string " << vetoStr
                                      << ", either a typo or this function needs to be updated";
  }
}

bool EleTkIsolFromCands::passTrackPreselection(const reco::TrackBase& trk, const TrkCuts& cuts) {
  return trk.hitPattern().numberOfValidHits() >= cuts.minHits &&
         trk.hitPattern().numberOfValidPixelHits() >= cuts.minPixelHits &&
         (trk.ptError() / trk.pt() < cuts.maxDPtPt || cuts.maxDPtPt < 0) && passQual(trk, cuts.allowedQualities) &&
         passAlgo(trk, cuts.algosToReject) && trk.pt() > cuts.minPt;
}

bool EleTkIsolFromCands::passMatchingToElectron(
    SimpleTrack const& trk, const TrkCuts& cuts, double eleEta, double elePhi, double eleVZ) {
  const float dR2 = reco::deltaR2(eleEta, elePhi, trk.eta, trk.phi);
  const float dEta = trk.eta - eleEta;
  const float dZ = eleVZ - trk.vz;

  return dR2 >= cuts.minDR2 && dR2 <= cuts.maxDR2 && std::abs(dEta) >= cuts.minDEta && std::abs(dZ) < cuts.maxDZ;
}

bool EleTkIsolFromCands::passQual(const reco::TrackBase& trk, const std::vector<reco::TrackBase::TrackQuality>& quals) {
  if (quals.empty())
    return true;

  for (auto qual : quals) {
    if (trk.quality(qual))
      return true;
  }

  return false;
}

bool EleTkIsolFromCands::passAlgo(const reco::TrackBase& trk,
                                  const std::vector<reco::TrackBase::TrackAlgorithm>& algosToRej) {
  return algosToRej.empty() || !std::binary_search(algosToRej.begin(), algosToRej.end(), trk.algo());
}
