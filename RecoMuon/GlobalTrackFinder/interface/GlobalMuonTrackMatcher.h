#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2007/01/16 17:02:46 $
 *  $Revision: 1.17 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

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

    bool matchPos(const TrackCand&,
		  const TrackCand&) const;
    
    /// check if two TSOS match at same surface
    double matchChiAtSurface(const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const;
    
    /// check if two Tracks match at IP
    double matchChiAtIP(const TrackCand&,
			const TrackCand&) const;

    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> convertToTSOS(const TrackCand&, const TrackCand&) const;

    bool matchPosAtSurface(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

    TrackCand matchMomAtIP(const TrackCand&, const std::vector<TrackCand>&) const;

  private:
    
    double theMaxChi2;
    double theMinP;
    double theMinPt;
    bool theMIMFlag;
    bool matchAtSurface_;
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;

    GlobalMuonMonitorInterface* dataMonitor;

    const MuonServiceProxy *theService;
};

#endif
