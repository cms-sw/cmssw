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

edm::ParameterSetDescription EleTkIsolFromCands::pSetDescript() {
  edm::ParameterSetDescription desc;
  desc.add("barrelCuts", TrkCuts::pSetDescript());
  desc.add("endcapCuts", TrkCuts::pSetDescript());
  return desc;
}

EleTkIsolFromCands::TrackTable EleTkIsolFromCands::preselectTracks(reco::TrackCollection const& tracks,
                                                                   TrkCuts const& cuts) {
  std::vector<float> pt;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> vz;
  pt.reserve(tracks.size());
  eta.reserve(tracks.size());
  phi.reserve(tracks.size());
  vz.reserve(tracks.size());

  for (auto const& trk : tracks) {
    auto trkPt = trk.pt();
    if (passTrackPreselection(trk, trkPt, cuts)) {
      pt.emplace_back(trkPt);
      eta.emplace_back(trk.eta());
      phi.emplace_back(trk.phi());
      vz.emplace_back(trk.vz());
    }
  }

  return {pt, std::move(eta), std::move(phi), std::move(vz)};
}

EleTkIsolFromCands::TrackTable EleTkIsolFromCands::preselectTracksFromCands(pat::PackedCandidateCollection const& cands,
                                                                            TrkCuts const& cuts,
                                                                            PIDVeto pidVeto) {
  std::vector<float> pt;
  std::vector<float> eta;
  std::vector<float> phi;
  std::vector<float> vz;
  pt.reserve(cands.size());
  eta.reserve(cands.size());
  phi.reserve(cands.size());
  vz.reserve(cands.size());

  for (auto const& cand : cands) {
    if (cand.hasTrackDetails() && cand.charge() != 0 && passPIDVeto(cand.pdgId(), pidVeto)) {
      const reco::Track& trk = cand.pseudoTrack();
      float trkPt = trk.pt();
      if (passTrackPreselection(trk, trkPt, cuts)) {
        pt.emplace_back(trkPt);
        eta.emplace_back(trk.eta());
        phi.emplace_back(trk.phi());
        vz.emplace_back(trk.vz());
      }
    }
  }

  return {pt, std::move(eta), std::move(phi), std::move(vz)};
}

EleTkIsolFromCands::Output EleTkIsolFromCands::operator()(const reco::TrackBase& eleTrk) {
  using namespace edm::soa::col;

  float ptSum = 0.;
  int nrTrks = 0;

  const float eleEta = eleTrk.eta();
  const float elePhi = eleTrk.phi();
  const float eleVz = eleTrk.vz();

  const bool isBarrelElectron = std::abs(eleEta) < 1.5;

  auto const& preselectedTracks = getPreselectedTracks(isBarrelElectron);
  auto const& cuts = isBarrelElectron ? cfg_.barrelCuts : cfg_.endcapCuts;

  for (auto const& trk : preselectedTracks) {
    const float dR2 = reco::deltaR2(eleEta, elePhi, trk.get<Eta>(), trk.get<Phi>());
    const float dEta = trk.get<Eta>() - eleEta;
    const float dZ = eleVz - trk.get<Vz>();

    if (dR2 >= cuts.minDR2 && dR2 <= cuts.maxDR2 && std::abs(dEta) >= cuts.minDEta && std::abs(dZ) < cuts.maxDZ) {
      ptSum += trk.get<Pt>();
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

bool EleTkIsolFromCands::passTrackPreselection(const reco::TrackBase& trk, float trackPt, const TrkCuts& cuts) {
  return trackPt > cuts.minPt && trk.numberOfValidHits() >= cuts.minHits &&
         trk.hitPattern().numberOfValidPixelHits() >= cuts.minPixelHits && passQual(trk, cuts.allowedQualities) &&
         passAlgo(trk, cuts.algosToReject) && (cuts.maxDPtPt < 0 || trk.ptError() / trackPt < cuts.maxDPtPt);
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

EleTkIsolFromCands::TrackTable const& EleTkIsolFromCands::getPreselectedTracks(bool isBarrel) {
  auto const& cuts = isBarrel ? cfg_.barrelCuts : cfg_.endcapCuts;
  auto& preselectedTracks = isBarrel ? preselectedTracksWithBarrelCuts_ : preselectedTracksWithEndcapCuts_;
  bool& tracksCached = isBarrel ? tracksCachedForBarrelCuts_ : tracksCachedForEndcapCuts_;

  if (!tracksCached) {
    preselectedTracks = tracks_ ? preselectTracks(*tracks_, cuts) : preselectTracksFromCands(*cands_, cuts, pidVeto_);
    tracksCached = true;
  }

  return preselectedTracks;
}
