#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"

/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker track
 *
 *  $Date: 2006/07/03 12:06:13 $
 *  $Revision: 1.3 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"


GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(const double& chi2, const MagneticField* field) {

  theMaxChi2 = chi2;
  theMinP = 2.5;
  theMinPt = 1.0;
  theField = field;
//  theTTrackBuilder = new TransientTrackBuilder(field);
  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09);
  theUpdator = new MuonUpdatorAtVertex(theVertexPos,theVertexErr,field);

}

//
/// choose the tk Track from a TrackCollection which has smallest Chi2 with
//   a given standalone Track

std::pair<bool, reco::TrackRef*> 
GlobalMuonTrackMatcher::matchOne(const reco::TrackRef* staT, const edm::Handle<reco::TrackCollection>& tkTs) const {
  bool hasMatchTk = false;
  reco::TrackRef* result = 0;
  double minChi2 = theMaxChi2;
  for (int position = 0; position < int(tkTs->size()); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    std::pair<bool,double> check = match(staT,&tkTRef);
    if (!check.first) continue;
    hasMatchTk = true;
    if (check.second < minChi2) { 
      minChi2 = check.second;
      result = &tkTRef;
    }
  } 
  return(std::pair<bool, reco::TrackRef*>(hasMatchTk, result));

}

//
// choose a vector of tk Tracks from a TrackCollection that has Chi2 less than
//    theMaxChi2, for a given standalone Track
//
std::vector<reco::TrackRef*>
GlobalMuonTrackMatcher::match(const reco::TrackRef* staT, const edm::Handle<reco::TrackCollection>& tkTs) const {

  std::vector<reco::TrackRef*> result;

  for (int position = 0; position < int(tkTs->size()); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    std::pair<bool,double> check = match(staT,&tkTRef);
    if (check.first) 
      result.push_back(&tkTRef);
  }
  return result;
}


//
// determine if two Tracks are compatible
// by comparing their TSOSs on the outer Tracker surface
//
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const reco::TrackRef* staTR, const reco::TrackRef* tkTR) const {

  reco::TransientTrack staTT(*staTR);  
  reco::TransientTrack tkTT(*tkTR);

  TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();
  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

// make sure the tk Track has enough momentum to reach muon chambers
  const GlobalVector& mom = outerTkTsos.globalMomentum();
   if ( mom.mag() < theMinP || mom.perp() < theMinPt ) 
     return std::pair<bool,double>(false,0);

// extrapolate innermost standalone TSOS to outer tk surface
  MuonVertexMeasurement vm = theUpdator->update(innerMuTsos);
  TrajectoryStateOnSurface tkTsosFromMu = vm.stateAtTracker();

// compare the TSOSs on outer tk surface
  return match(tkTsosFromMu,outerTkTsos);

}

//
// determine if two TSOSs are compatible, they should be on same surface,
//    usually the outer tk surface
// 
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const TrajectoryStateOnSurface& tsos1, 
                              const TrajectoryStateOnSurface& tsos2) const {

  AlgebraicVector v(tsos1.globalParameters().vector() - tsos2.globalParameters().vector());
  AlgebraicSymMatrix m(tsos1.curvilinearError().matrix() + tsos2.curvilinearError().matrix());
  int ierr;
  m.invert(ierr);
  // if (ierr != 0) throw exception;
  double est = m.similarity(v);

  return ( est > theMaxChi2 ) ? std::pair<bool,double>(false,est) : std::pair<bool,double>(true,est);

}

