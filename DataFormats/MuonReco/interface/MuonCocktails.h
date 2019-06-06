#ifndef MuonReco_MuonCocktails_h
#define MuonReco_MuonCocktails_h

/** \class MuonCocktails
 *
 *  Set of functions that select among the different track refits
 *  based on the fit quality, in order to achieve optimal resolution.
 *
 *  \author Piotr Traczyk,
 *   modified by Raffaella Radogna
 */

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

namespace muon {

  reco::Muon::MuonTrackTypePair tevOptimized(const reco::TrackRef& combinedTrack,
                                             const reco::TrackRef& trackerTrack,
                                             const reco::TrackRef& tpfmsTrack,
                                             const reco::TrackRef& pickyTrack,
                                             const reco::TrackRef& dytTrack,
                                             const double ptThreshold = 200.,
                                             const double tune1 = 17.,
                                             const double tune2 = 40.,
                                             const double dptcut = 0.25);

  // Version for convenience. (NB: can be used with pat::Muon, even
  // with embedded tracks, equally conveniently!)
  inline reco::Muon::MuonTrackTypePair tevOptimized(const reco::Muon& muon,
                                                    const double ptThreshold = 200.,
                                                    const double tune1 = 17.,
                                                    const double tune2 = 40.,
                                                    const double dptcut = 0.25) {
    return tevOptimized(muon.globalTrack(),
                        muon.innerTrack(),
                        muon.tpfmsTrack(),
                        muon.pickyTrack(),
                        muon.dytTrack(),
                        ptThreshold,
                        tune1,
                        tune2,
                        dptcut);
  }

  reco::TrackRef getTevRefitTrack(const reco::TrackRef& combinedTrack, const reco::TrackToTrackMap& map);

  // The cocktail used as the soon-to-be-old default momentum
  // assignment for the reco::Muon.
  reco::Muon::MuonTrackTypePair sigmaSwitch(const reco::TrackRef& combinedTrack,
                                            const reco::TrackRef& trackerTrack,
                                            const double nSigma = 2.,
                                            const double ptThreshold = 200.);

  // Convenience version of the above.
  inline reco::Muon::MuonTrackTypePair sigmaSwitch(const reco::Muon& muon,
                                                   const double nSigma = 2.,
                                                   const double ptThreshold = 200.) {
    return muon::sigmaSwitch(muon.globalTrack(), muon.innerTrack(), nSigma, ptThreshold);
  }

  // "Truncated muon reconstructor": the first cocktail, between just
  // tracker-only and TPFMS. Similar to tevOptimized.
  reco::Muon::MuonTrackTypePair TMR(const reco::TrackRef& trackerTrack,
                                    const reco::TrackRef& fmsTrack,
                                    const double tune = 4.);

  double trackProbability(const reco::TrackRef track);
}  // namespace muon

#endif
