#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/07/02 03:00:36 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Handle.h"

class TrajectoryStateOnSurface;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {

  public:

    /// constructor
    GlobalMuonTrackMatcher(double chi2);

    /// destructor
    virtual ~GlobalMuonTrackMatcher() {};

    ///
    std::pair<bool, reco::TrackRef> match(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>&) const;

    ///
    std::pair<bool,double> match(const reco::TrackRef&, const reco::TrackRef&) const; 

    ///
    std::pair<bool,double> match(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

  private:

    double theMaxChi2;

};

#endif
