#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/09/20 16:38:09 $
 *  $Revision: 1.15 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

class TrajectoryStateOnSurface;
class MuonServiceProxy;
class Trajectory;
class MuonUpdatorAtVertex;


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
    
    /// check if two TrackRefs match
    std::pair<bool,double> match(const TrackCand&,
                                 const TrackCand&) const;
    
    /// check if two TSOS match
    std::pair<bool,double> match(const TrajectoryStateOnSurface&, 
                                 const TrajectoryStateOnSurface&) const;
    
  private:
    
    double theMaxChi2;
    double theMinP;
    double theMinPt;
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;
    MuonUpdatorAtVertex* theUpdator;

    const MuonServiceProxy *theService;
};

#endif
