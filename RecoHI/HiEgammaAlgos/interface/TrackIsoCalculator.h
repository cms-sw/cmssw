#ifndef HiEgammaAlgos_TrackIsoCalculator_h
#define HiEgammaAlgos_TrackIsoCalculator_h

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class TrackIsoCalculator {
public:
  TrackIsoCalculator(reco::TrackCollection const& trackCollection, std::string const& trackQuality);

  /// Return the tracker energy in a cone around the photon
  double getTrackIso(reco::Photon const& clus, const double i, const double threshold, const double innerDR = 0);
  /// Return the background-subtracted tracker energy in a cone around the photon
  double getBkgSubTrackIso(reco::Photon const& clus, const double i, const double threshold, const double innerDR = 0);

private:
  reco::TrackCollection const& recCollection_;
  std::string const& trackQuality_;
};

#endif
