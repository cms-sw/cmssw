#ifndef GlobalTrackFinder_GlobalMuonTrackMatcher_H
#define GlobalTrackFinder_GlobalMuonTrackMatcher_H

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/08/03 17:28:50 $
 *  $Revision: 1.8 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

class TrajectoryStateOnSurface;
class MuonUpdatorAtVertex;
class MagneticField;
class GlobalTrackingGeometry;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonTrackMatcher {

  public:

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
    std::pair<bool, reco::TrackRef> matchOne(const reco::TrackRef&, 
                                             const edm::Handle<reco::TrackCollection>&) const;

    /// choose all that has chi2 less than MaxChi2
    std::vector<reco::TrackRef> match(const reco::TrackRef&, 
                                      const edm::Handle<reco::TrackCollection>&) const;

    /// choose all that has chi2 less than MaxChi2
    std::vector<reco::TrackRef> match(const reco::TrackRef&, 
                                      const std::vector<reco::TrackRef>&) const;

    /// check if two trackRefs are match
    std::pair<bool,double> match(const reco::TrackRef&,
                                 const reco::TrackRef&) const;

    /// check if two tracks are match
    std::pair<bool,double> match(const reco::Track&, 
                                 const reco::Track&) const; 

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
