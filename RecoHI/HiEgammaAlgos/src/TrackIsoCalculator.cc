#include "RecoHI/HiEgammaAlgos/interface/TrackIsoCalculator.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace reco;
using namespace std;

TrackIsoCalculator::TrackIsoCalculator(reco::TrackCollection const& trackCollection, std::string const& trackQuality)
    : recCollection_{trackCollection}, trackQuality_{trackQuality} {}

double TrackIsoCalculator::getTrackIso(reco::Photon const& cluster,
                                       const double x,
                                       const double threshold,
                                       const double innerDR) {
  double totalPt = 0;

  for (auto const& recTrack : recCollection_) {
    bool goodtrack = recTrack.quality(reco::TrackBase::qualityByName(trackQuality_));
    if (!goodtrack)
      continue;

    double pt = recTrack.pt();
    double dR2 = reco::deltaR2(cluster, recTrack);
    if (dR2 >= (0.01 * x * x))
      continue;
    if (dR2 < innerDR * innerDR)
      continue;
    if (pt > threshold)
      totalPt = totalPt + pt;
  }

  return totalPt;
}

double TrackIsoCalculator::getBkgSubTrackIso(reco::Photon const& cluster,
                                             const double x,
                                             const double threshold,
                                             const double innerDR) {
  double SClusterEta = cluster.eta();
  double totalPt = 0.0;

  for (auto const& recTrack : recCollection_) {
    bool goodtrack = recTrack.quality(reco::TrackBase::qualityByName(trackQuality_));
    if (!goodtrack)
      continue;

    double pt = recTrack.pt();
    if (std::abs(recTrack.eta() - SClusterEta) >= 0.1 * x)
      continue;
    if (reco::deltaR2(cluster, recTrack) < innerDR * innerDR)
      continue;

    if (pt > threshold)
      totalPt = totalPt + pt;
  }

  double Tx = getTrackIso(cluster, x, threshold, innerDR);
  double CTx = (Tx - totalPt / 40.0 * x) * (1 / (1 - x / 40.));

  return CTx;
}
