#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2007/02/16 23:32:32 $
 *  $Revision: 1.21 $
 *
 *  \author Chang Liu           Purdue University
 *  \author Adam Everett        Purdue University
 *  \author Norbert Neumeister  Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"

class TrajectoryStateOnSurface;
class MuonServiceProxy;
class Trajectory;
class GlobalMuonMonitorInterface;

namespace edm {class ParameterSet;}


//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {

  public:

    typedef std::pair<const Trajectory*,reco::TrackRef> TrackCand;

    /// constructor
    GlobalMuonTrackMatcher(const edm::ParameterSet& par,
                           const MuonServiceProxy*);

    /// destructor
    virtual ~GlobalMuonTrackMatcher();
    
    /// choose the track with smallest matching-chi2
    std::pair<bool, TrackCand> matchOne(const TrackCand&, 
					const std::vector<TrackCand>&) const;
    
    /// choose all tracks with a matching-chi2 less than MaxChi2
    std::vector<TrackCand> match(const TrackCand&, 
				 const std::vector<TrackCand>&) const;
    
    /// check if two TrackRefs match at tracker surface
    std::pair<bool,double> matchChi(const TrackCand&,
                                    const TrackCand&) const;

    /// check position of two tracks
    bool matchPos(const TrackCand&,
		  const TrackCand&) const;
    
    /// check if two TSOS match at same surface
    double matchChiAtSurface(const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const;
    
    /// check if two Tracks match at IP
    double matchChiAtIP(const TrackCand&,
			const TrackCand&) const;

    /// propagate two tracks to a common surface
    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> convertToTSOS(const TrackCand&, 
                                                                                const TrackCand&) const;

    /// compare global positions of track candidates
    bool matchPosAtSurface(const TrajectoryStateOnSurface&, 
                           const TrajectoryStateOnSurface&) const;

    /// compare global directions of track candidates
    TrackCand matchMomAtIP(const TrackCand&, 
                           const std::vector<TrackCand>&) const;

  private:
    
    double theMaxChi2;
    double theMinP;
    double theMinPt;
    double theDeltaEta;
    double theDeltaPhi;
    bool theMIMFlag;
    bool matchAtSurface_;

    GlobalMuonMonitorInterface* dataMonitor;

    const MuonServiceProxy *theService;
    std::string theOutPropagatorName;
};

#endif
