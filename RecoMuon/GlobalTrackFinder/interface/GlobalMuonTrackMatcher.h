#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: $
 *  $Revision:  $
 *  \author Chang Liu  - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Handle.h"


class TrajectoryStateOnSurface;

class GlobalMuonTrackMatcher {
public:
  /// constructor
  GlobalMuonTrackMatcher(const double& chi2);

  /// destructor
  virtual ~GlobalMuonTrackMatcher(){};

  std::pair<bool, reco::TrackRef> match(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>&);

  std::pair<bool,double> match(const reco::TrackRef&, const reco::TrackRef&); 

  std::pair<bool,double> match(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&);

private:
  double theMaxChi2;

};
#endif
