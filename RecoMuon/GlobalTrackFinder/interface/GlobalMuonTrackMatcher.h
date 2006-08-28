#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/08/28 19:59:33 $
 *  $Revision: 1.11 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

class TrajectoryStateOnSurface;
class MagneticField;
class GlobalTrackingGeometry;
class Trajectory;
class MuonUpdatorAtVertex;

namespace edm {class EventSetup;}


//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {

  public:

    typedef std::pair<Trajectory*,reco::TrackRef> TrackCand;

    /// constructor
    GlobalMuonTrackMatcher(double chi2,
                           const edm::ESHandle<MagneticField>,
                           MuonUpdatorAtVertex*);

    GlobalMuonTrackMatcher(double chi2);

    /// destructor
    virtual ~GlobalMuonTrackMatcher() {}
    
    /// set event setup
    void setES(const edm::EventSetup&);

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
    edm::ESHandle<MagneticField> theField;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

};

#endif
