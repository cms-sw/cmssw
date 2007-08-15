#ifndef GlobalTrackingTools_GlobalMuonTrackMatcher_H
#define GlobalTrackingTools_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher 
 *
 * Match a standalone muon track with the most compatible tracker tracks
 * in a TrackCollection.  GlobalMuonTrackMatcher is used during global
 * muon reconstruction to check the compatability of a tracker track
 * with a standalone muon track.  The compatability is determined based
 * on a chi2 comparison of the local parameters of the two corresponding
 * TrajectoryStateOnSurface, with the surface being the tracker outer
 * bound.  If the comparison of local parameters fails to yield any
 * matches, then it makes a comparison of the TSOS global positon.  If
 * there is still no match, then it matches the standalone muon with the
 * tracker track that is closest in eta-phi space.
 *
 *
 *  $Date: $
 *  $Revision: $
 *
 *  \author Chang Liu           Purdue University
 *  \author Adam Everett        Purdue University
 *  \author Norbert Neumeister  Purdue University
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"

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
    
    /// choose the one track with smallest matching-chi2
    std::pair<bool, TrackCand> matchOne(const TrackCand&, 
					const std::vector<TrackCand>&) const;
    
    /// choose all tracks with a matching-chi2 less than MaxChi2
    std::vector<TrackCand> match(const TrackCand&, 
				 const std::vector<TrackCand>&) const;
    
    /// check if two TrackRefs have matching local parameters on tracker surface
    std::pair<bool,double> matchChi(const TrackCand&,
                                    const TrackCand&) const;

    /// check position of two tracks
    bool matchPos(const TrackCand&,
		  const TrackCand&) const;

    /// compare global directions of track candidates
    TrackCand matchMomAtIP(const TrackCand&, 
                           const std::vector<TrackCand>&) const;
    
    /// check if two TSOS match at same surface
    double matchChiAtSurface(const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const;

 private:
    /// propagate two tracks to a common surface - the tracker outer bound
    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> convertToTSOS(const TrackCand&, const TrackCand&) const;
    
    /// compare global positions of track candidates
    bool matchPosAtSurface(const TrajectoryStateOnSurface&, 
                           const TrajectoryStateOnSurface&) const;


    /// check that two TSOSs are on the same plane
    bool samePlane(const TrajectoryStateOnSurface& tsos1,
		   const TrajectoryStateOnSurface& tsos2) const;

  private:
    
    double theMaxChi2;
    double theMinP;
    double theMinPt;
    double theDeltaEta;
    double theDeltaPhi;

    const MuonServiceProxy *theService;
    std::string theOutPropagatorName;

};

#endif
