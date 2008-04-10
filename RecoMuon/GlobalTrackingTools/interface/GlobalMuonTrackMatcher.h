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
 *  $Date: 2008/02/05 22:21:20 $
 *  $Revision: 1.3 $
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
    
 private:

  double match_R_IP(const TrackCand&, const TrackCand&) const;
  double match_D(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
  double match_d(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
  double match_Rmom(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
  double match_Rpos(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;
  double match_ChiAtSurface(const TrajectoryStateOnSurface& , const TrajectoryStateOnSurface& ) const;

  std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSTk(const TrackCand&,const TrackCand& ) const;
  std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSMuHit(const TrackCand&,const TrackCand& ) const;
  std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface> convertToTSOSTkHit(const TrackCand&,const TrackCand& ) const;
  bool samePlane(const TrajectoryStateOnSurface&,const TrajectoryStateOnSurface&) const;


  private:
    
    double theMaxChi2;
    double theMinP;
    double theMinPt;
    double theDeltaD;
    double theDeltaR;

    const MuonServiceProxy *theService;
    std::string theOutPropagatorName;

};

#endif
