#ifndef MuonReco_MuonCocktails_h
#define MuonReco_MuonCocktails_h

/** \class MuonCocktails
 *
 *  Set of functions that select among the different track refits
 *  based on the fit quality, in order to achieve optimal resolution.
 *
 *  $Date: 2012/10/04 09:42:00 $
 *  $Revision: 1.10 $
 *  \author Piotr Traczyk
 */

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

namespace muon {
  
  reco::Muon::MuonTrackTypePair tevOptimized(const reco::TrackRef& combinedTrack,
					     const reco::TrackRef& trackerTrack,
					     const reco::TrackRef& tpfmsTrack,
					     const reco::TrackRef& pickyTrack,
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
			ptThreshold,
			tune1,
			tune2,
			dptcut);  
  }

  reco::TrackRef getTevRefitTrack(const reco::TrackRef& combinedTrack,
				  const reco::TrackToTrackMap& map);
  
  // The next two versions of tevOptimized are for backward
  // compatibility; TrackToTrackMaps are to be removed from the
  // EventContent, so these versions will go away (along with the
  // helper getter function) after a deprecation period. Since they
  // are just for backward compatibility and not for new code, we
  // don't bother to expose the tune parameters.

  inline reco::Muon::MuonTrackTypePair tevOptimized(const reco::TrackRef& combinedTrack,
						    const reco::TrackRef& trackerTrack,
						    const reco::TrackToTrackMap& tevMap1,
						    const reco::TrackToTrackMap& tevMap2,
						    const reco::TrackToTrackMap& tevMap3,
						    const double ptThreshold = 200.,
						    const double tune1 = 17.,
						    const double tune2 = 40.,
						    const double dptcut = 0.25) {
    return tevOptimized(combinedTrack,
			trackerTrack,
			getTevRefitTrack(combinedTrack, tevMap2),
			getTevRefitTrack(combinedTrack, tevMap3),
			ptThreshold,
			tune1,
			tune2,
			dptcut);
  }
  
  inline reco::Muon::MuonTrackTypePair tevOptimized(const reco::Muon& muon,
						    const reco::TrackToTrackMap& tevMap1,
						    const reco::TrackToTrackMap& tevMap2,
						    const reco::TrackToTrackMap& tevMap3 ) {
    return tevOptimized(muon.combinedMuon(),
			muon.track(),
			getTevRefitTrack(muon.combinedMuon(), tevMap2),
			getTevRefitTrack(muon.combinedMuon(), tevMap3));
  }
  
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
    return muon::sigmaSwitch(muon.globalTrack(),
			     muon.innerTrack(),
			     nSigma,
			     ptThreshold);
  }

  // "Truncated muon reconstructor": the first cocktail, between just
  // tracker-only and TPFMS. Similar to tevOptimized.
  reco::Muon::MuonTrackTypePair TMR(const reco::TrackRef& trackerTrack,
				    const reco::TrackRef& fmsTrack,
				    const double tune=4.);
  
  double trackProbability(const reco::TrackRef track);
}

#endif
