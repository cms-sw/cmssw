/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker tracks
 *
 *  $Date: 2006/08/28 19:32:59 $
 *  $Revision: 1.21 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 *  \author Adam Everett - Purdue University
 */

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

using namespace std;
using namespace edm;
//
// constructor
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(double chi2, 
                                               const edm::ESHandle<MagneticField> field, 
                                               MuonUpdatorAtVertex* updator) {

  theMaxChi2 = chi2;
  theMinP = 2.5;
  theMinPt = 1.0;
  theField = field;
  theUpdator = updator;

}


//
//
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(double chi2) {

  theMaxChi2 = chi2;
  theMinP = 2.5;
  theMinPt = 1.0;
  theUpdator = new MuonUpdatorAtVertex();

}


//
//
//
void GlobalMuonTrackMatcher::setES(const edm::EventSetup& setup) {

  setup.get<IdealMagneticFieldRecord>().get(theField);
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  theUpdator->setES(setup);

}


//
// choose the tracker Track from a TrackCollection which has smallest chi2 with
// a given standalone Track
//
pair<bool, GlobalMuonTrackMatcher::TrackCand> 
GlobalMuonTrackMatcher::matchOne(const TrackCand& staCand, 
                                 const vector<TrackCand>& tkTs) const {

  bool hasMatchTk = false;
  TrackCand result = staCand;
  double minChi2 = theMaxChi2;
  
  TrajectoryStateOnSurface innerMuTsos;
  if(staCand.first == 0) {
    reco::TransientTrack staTT(staCand.second,&*theField,theTrackingGeometry);
    innerMuTsos = staTT.innermostMeasurementState();
  } else {
    innerMuTsos = staCand.first->firstMeasurement().updatedState();
  }
  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);
  if ( !tkTsosFromMu.isValid() ) return pair<bool,TrackCand>(false,result);
 
  for(vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    TrajectoryStateOnSurface outerTkTsos;
    if((*is).first == 0) {
      reco::TrackRef tkTRef(is->second);
      reco::TransientTrack tkTT(tkTRef,&*theField,theTrackingGeometry);
      // make sure the tracker Track has enough momentum to reach muon chambers
      const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
      if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;
      outerTkTsos = tkTT.outermostMeasurementState();
    } else {
      outerTkTsos = (*is).first->lastMeasurement().updatedState();
    }
    
    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
    if ( !tkTsosFromTk.isValid() ) continue; 
    
    pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);
    
    if (!check.first) continue;
    
    if (check.second < minChi2) { 
      hasMatchTk = true;
      minChi2 = check.second;
      result = (*is);
    } 
  }     

  return pair<bool, TrackCand>(hasMatchTk, result);

}


//
// choose a vector of tracker Tracks from a TrackCollection that has Chi2 
// less than theMaxChi2, for a given standalone Track
//
vector<GlobalMuonTrackMatcher::TrackCand>
GlobalMuonTrackMatcher::match(const TrackCand& staCand, 
                              const std::vector<TrackCand>& tkTs) const {
  
  vector<TrackCand> result;

  TrajectoryStateOnSurface innerMuTsos;

  if(staCand.first == 0) {
    reco::TransientTrack staTT(staCand.second,&*theField,theTrackingGeometry);
    innerMuTsos = staTT.innermostMeasurementState();
  } else {
    innerMuTsos = staCand.first->firstMeasurement().updatedState();
  }
  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);
  if ( !tkTsosFromMu.isValid() ) return result;
 
  for(vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    TrajectoryStateOnSurface outerTkTsos;
    if((*is).first == 0) {
      reco::TrackRef tkTRef(is->second);
      reco::TransientTrack tkTT(tkTRef,&*theField,theTrackingGeometry);
      // make sure the tracker Track has enough momentum to reach muon chambers
      const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
      if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue; 
      outerTkTsos = tkTT.outermostMeasurementState();
    } else {
      outerTkTsos = (*is).first->lastMeasurement().updatedState();
    }
    
    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);    
    if ( !tkTsosFromTk.isValid() ) continue; 
    
    pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);
    
    if ( check.first ) result.push_back(*is);

  }
  
  return result;
  
}


//
// determine if two TrackRefs are compatible
// by comparing their TSOSs on the outer Tracker surface
//
pair<bool,double> 
GlobalMuonTrackMatcher::match(const TrackCand& staCand, 
                              const TrackCand& tkCand) const {

  TrajectoryStateOnSurface innerMuTsos;  
  TrajectoryStateOnSurface outerTkTsos;

  if(staCand.first == 0) {
    reco::TransientTrack staT(staCand.second,&*theField,theTrackingGeometry);  
    innerMuTsos = staT.innermostMeasurementState();
  } else {
    innerMuTsos = staCand.first->firstMeasurement().updatedState();
  }
  
  if(tkCand.first == 0) {
    reco::TransientTrack tkT(tkCand.second,&*theField,theTrackingGeometry);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt )
      return pair<bool,double>(false,0);
    outerTkTsos = tkT.outermostMeasurementState();
  } else {
    outerTkTsos = tkCand.first->lastMeasurement().updatedState();
  }
  
  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);
  if ( !tkTsosFromMu.isValid() ) return pair<bool,double>(false,0.);
 
  // extrapolate outermost tracker measurement TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
  if ( !tkTsosFromTk.isValid() ) return pair<bool,double>(false,0.);
  // compare the TSOSs on outer tracker surface
  return match(tkTsosFromMu,tkTsosFromTk);
  
}


//
// determine if two TSOSs are compatible, they should be on same surface
// 
pair<bool,double> 
GlobalMuonTrackMatcher::match(const TrajectoryStateOnSurface& tsos1, 
                              const TrajectoryStateOnSurface& tsos2) const {

  AlgebraicVector v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix m(tsos1.localError().matrix() + tsos2.localError().matrix());
  int ierr;
  m.invert(ierr);
  // if (ierr != 0) throw exception;
  double est = m.similarity(v);
  
  bool goodChi =  ( est > theMaxChi2 ) ? false : true;

  double d((tsos1.globalPosition()- tsos2.globalPosition()).mag());
  double dx(fabs(tsos1.globalPosition().x() - tsos2.globalPosition().x()));
  double dy(fabs(tsos1.globalPosition().y() - tsos2.globalPosition().y()));
  double dz(fabs(tsos1.globalPosition().z() - tsos2.globalPosition().z()));

  float dd = 5.0;
  bool goodCoords = ( (dx < dd) && (dy < dd) ) ? true : false;
  bool good = ( goodChi || goodCoords ) ? true : false;

  return pair<bool,double>(good,est);

  //return ( est > theMaxChi2 ) ? pair<bool,double>(false,est) : pair<bool,double>(true,est);

}

