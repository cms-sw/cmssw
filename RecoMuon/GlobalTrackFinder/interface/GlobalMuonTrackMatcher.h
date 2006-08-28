#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/08/09 16:40:28 $
 *  $Revision: 1.9 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

class TrajectoryStateOnSurface;
class MagneticField;
class GlobalTrackingGeometry;
class Trajectory;

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
    
    /// set eventsetup
    void setES(const edm::EventSetup&);

    /// choose one that with smallest chi2
    std::pair<bool, TrackCand> matchOne(TrackCand&, 
					std::vector<TrackCand>&) const;
    
    /// choose all that has chi2 less than MaxChi2
    std::vector<TrackCand> match(TrackCand&, 
				 std::vector<TrackCand>&) const;
    
    /// check if two trackRefs are match
    std::pair<bool,double> match(TrackCand&,
                                 TrackCand&) const;
    
    /// check if two TSOS are match
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
