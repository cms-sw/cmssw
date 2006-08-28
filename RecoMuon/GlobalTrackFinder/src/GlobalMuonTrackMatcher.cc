/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker tracks
 *
 *  $Date: 2006/08/10 15:19:43 $
 *  $Revision: 1.19 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

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
pair<bool, reco::TrackRef> 
GlobalMuonTrackMatcher::matchOne(const reco::TrackRef& staT, 
                                 const edm::Handle<reco::TrackCollection>& tkTs) const {

  bool hasMatchTk = false;
  reco::TrackRef result;
  double minChi2 = theMaxChi2;
  
  reco::TransientTrack staTT(staT,&*theField,theTrackingGeometry);
  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);

  for (unsigned int position = 0; position < tkTs->size(); ++position) {
    reco::TrackRef tkTRef(tkTs,position);
    reco::TransientTrack tkTT(tkTRef,&*theField,theTrackingGeometry);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;

    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();
    
    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    TrajectoryStateOnSurface tkTsosFromTk = 
      theUpdator->stateAtTracker(outerTkTsos);

    pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);

    if (!check.first) continue;

    if (check.second < minChi2) { 
      hasMatchTk = true;
      minChi2 = check.second;
      result = tkTRef;
    } 
  }     

  return pair<bool, reco::TrackRef>(hasMatchTk, result);

}


//
// choose a vector of tracker Tracks from a TrackCollection that has Chi2 less than
// theMaxChi2, for a given standalone Track
//
vector<reco::TrackRef>
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, 
                              const edm::Handle<reco::TrackCollection>& tkTs) const {

  vector<reco::TrackRef> result;

  reco::TransientTrack staTT(staT,&*theField,theTrackingGeometry);

  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = 
    theUpdator->stateAtTracker(innerMuTsos);
 
  for (unsigned int position = 0; position < tkTs->size(); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    reco::TransientTrack tkTT(tkTRef,&*theField,theTrackingGeometry);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;
    
    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();
    
    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
    
    pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);
    if (check.first) result.push_back(tkTRef);
  }
  
  return result;
  
}


//
//
//
vector<reco::TrackRef>
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, 
                              const vector<reco::TrackRef>& tkTs) const {

  vector<reco::TrackRef> result;

  reco::TransientTrack staTT(staT,&*theField,theTrackingGeometry);

  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);

  if ( !tkTsosFromMu.isValid() ) return result;
   
  //cout << "M: " << tkTsosFromMu.globalPosition() << " "
  //              << tkTsosFromMu.globalMomentum() << endl;

  for (vector<reco::TrackRef>::const_iterator tkTRef = tkTs.begin();
       tkTRef != tkTs.end(); tkTRef++) {

    reco::TransientTrack tkTT(*tkTRef, &*theField,theTrackingGeometry);

    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;
    
    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();
    
    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);

    if ( !tkTsosFromTk.isValid() ) continue; 

    //cout << "T: " << tkTsosFromTk.globalPosition() << " "
    //              << tkTsosFromTk.globalMomentum() << endl;

    pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);

    double d((tkTsosFromMu.globalPosition()- tkTsosFromTk.globalPosition()).mag());
    double dx(fabs(tkTsosFromMu.globalPosition().x() - tkTsosFromTk.globalPosition().x()));
    double dy(fabs(tkTsosFromMu.globalPosition().y() - tkTsosFromTk.globalPosition().y()));
    double dz(fabs(tkTsosFromMu.globalPosition().z() - tkTsosFromTk.globalPosition().z()));

    float dd = 5.0;
    bool goodCoords = ( (dx < dd) && (dy < dd) ) ? true : false;
    bool good = ( check.first || goodCoords ) ? true : false;

    //cout << "Match: " << good << " " << check.second << " " 
    //                  << d << " " << dx << " " << dy << " " << dz << endl;

    if ( good ) result.push_back(*tkTRef);
  }

  return result;

}


//
// determine if two TrackRefs are compatible
// by comparing their TSOSs on the outer Tracker surface
//
pair<bool,double> 
GlobalMuonTrackMatcher::match(const reco::TrackRef& sta, 
                              const reco::TrackRef& tk) const {

  reco::TransientTrack staT(sta,&*theField,theTrackingGeometry);  
  reco::TransientTrack tkT(tk,&*theField,theTrackingGeometry);
  
  // make sure the tracker Track has enough momentum to reach muon chambers
  const GlobalVector& mom = tkT.impactPointState().globalMomentum();
  if ( mom.mag() < theMinP || mom.perp() < theMinPt )
    return pair<bool,double>(false,0);
  
  TrajectoryStateOnSurface outerTkTsos = tkT.outermostMeasurementState();
  
  TrajectoryStateOnSurface innerMuTsos = staT.innermostMeasurementState();
  
  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);
  
  // extrapolate outermost tracker measurement TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
  
  // compare the TSOSs on outer tracker surface
  return match(tkTsosFromMu,tkTsosFromTk);
  
}

//
// determine if two Tracks are compatible
// by comparing their TSOSs on the outer Tracker surface
//
pair<bool,double> 
GlobalMuonTrackMatcher::match(const reco::Track& sta, 
                              const reco::Track& tk) const {

  reco::TransientTrack staT(sta,&*theField,theTrackingGeometry);  
  reco::TransientTrack tkT(tk,&*theField,theTrackingGeometry);
  //FIXME: caution! no Track * stored in TransientTrack.

  // make sure the tracker Track has enough momentum to reach muon chambers
  const GlobalVector& mom = tkT.impactPointState().globalMomentum();
  if ( mom.mag() < theMinP || mom.perp() < theMinPt )
    return pair<bool,double>(false,0);
  
  TrajectoryStateOnSurface outerTkTsos = tkT.outermostMeasurementState();
  
  TrajectoryStateOnSurface innerMuTsos = staT.innermostMeasurementState();
  
  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);
  
  // extrapolate outermost tracker measurement TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
  
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

  return ( est > theMaxChi2 ) ? pair<bool,double>(false,est) : pair<bool,double>(true,est);

}

