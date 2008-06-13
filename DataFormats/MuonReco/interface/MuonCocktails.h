#ifndef MuonReco_MuonCocktails_h
#define MuonReco_MuonCocktails_h

/** \class MuonCocktails
 *
 *  Set of functions that select among the different track refits
 *  based on the fit quality, in order to achieve optimal resolution.
 *
 *  $Date: $
 *  $Revision: $
 *  \author Piotr Traczyk
 */

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

namespace muon {

  reco::TrackRef tevOptimized( const reco::TrackRef& combinedTrack,
			       const reco::TrackRef& trackerTrack,
			       const reco::TrackToTrackMap tevMap1,
			       const reco::TrackToTrackMap tevMap2,
			       const reco::TrackToTrackMap tevMap3 );

  reco::TrackRef tevOptimizedOld( const reco::TrackRef& combinedTrack,
				  const reco::TrackRef& trackerTrack,
				  const reco::TrackToTrackMap tevMap1,
				  const reco::TrackToTrackMap tevMap2,
				  const reco::TrackToTrackMap tevMap3 );

  reco::TrackRef tevOptimized( const reco::Muon& muon,
			       const reco::TrackToTrackMap tevMap1,
			       const reco::TrackToTrackMap tevMap2,
			       const reco::TrackToTrackMap tevMap3 );

  reco::TrackRef tevOptimizedOld( const reco::Muon& muon,
				  const reco::TrackToTrackMap tevMap1,
				  const reco::TrackToTrackMap tevMap2,
				  const reco::TrackToTrackMap tevMap3 );

  double trackProbability(const reco::TrackRef track);

}
#endif
