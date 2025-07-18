#include "RecoEgamma/EgammaIsolationAlgos/interface/SuperclusTkIsolFromCands.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

SuperclusTkIsolFromCands::Output SuperclusTkIsolFromCands::operator()(const reco::SuperCluster& sc,
                                                                      const math::XYZPoint& vtx) {
  using namespace edm::soa::col;

  float ptSum = 0.;
  int nrTrks = 0;

  const float scEta = sc.eta();
  const float scPhi = sc.phi();
  const float vtxVz = vtx.z();

  const bool isBarrelSC = std::abs(scEta) < 1.5;

  auto const& preselectedTracks = getPreselectedTracks(isBarrelSC);
  auto const& cuts = isBarrelSC ? cfg_.barrelCuts : cfg_.endcapCuts;

  for (auto const& trk : preselectedTracks) {
    const float dR2 = reco::deltaR2(scEta, scPhi, trk.get<Eta>(), trk.get<Phi>());
    const float dEta = trk.get<Eta>() - scEta;
    const float dZ = vtxVz - trk.get<Vz>();

    if (dR2 >= cuts.minDR2 && dR2 <= cuts.maxDR2 && std::abs(dEta) >= cuts.minDEta && std::abs(dZ) < cuts.maxDZ) {
      ptSum += trk.get<Pt>();
      nrTrks++;
    }
  }

  return {nrTrks, ptSum};
}
