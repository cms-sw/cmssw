/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker tracks
 *
 *  $Date: 2006/07/20 13:03:27 $
 *  $Revision: 1.9 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 */

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

//
// constructor
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(double chi2, const MagneticField* field) {

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
// choose the tracker Track from a TrackCollection which has smallest chi2 with
// a given standalone Track
std::pair<bool, reco::TrackRef> 
GlobalMuonTrackMatcher::matchOne(const reco::TrackRef& staT, 
                                 const edm::Handle<reco::TrackCollection>& tkTs) const {

  bool hasMatchTk = false;
  reco::TrackRef* result = 0;
  double minChi2 = theMaxChi2;

  reco::TransientTrack staTT(staT);

  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

// extrapolate innermost standalone TSOS to outer tracker surface
  MuonVertexMeasurement vm = theUpdator->update(innerMuTsos);
  TrajectoryStateOnSurface tkTsosFromMu = vm.stateAtTracker();

  for (unsigned int position = 0; position < tkTs->size(); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    reco::TransientTrack tkTT(tkTRef);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;

    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();

    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    vm = theUpdator->update(outerTkTsos);
    TrajectoryStateOnSurface tkTsosFromTk = vm.stateAtTracker();

    std::pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);

    if (!check.first) continue;
    hasMatchTk = true;
    if (check.second < minChi2) { 
      minChi2 = check.second;
      result = &tkTRef;
    }
  } 

  return(std::pair<bool, reco::TrackRef>(hasMatchTk, *result));

}


//
// choose a vector of tracker Tracks from a TrackCollection that has Chi2 less than
// theMaxChi2, for a given standalone Track
//
std::vector<reco::TrackRef>
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, 
                              const edm::Handle<reco::TrackCollection>& tkTs) const {

  std::vector<reco::TrackRef> result;

  reco::TransientTrack staTT(staT);

  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

// extrapolate innermost standalone TSOS to outer tracker surface
  MuonVertexMeasurement vm = theUpdator->update(innerMuTsos);
  TrajectoryStateOnSurface tkTsosFromMu = vm.stateAtTracker();

  for (unsigned int position = 0; position < tkTs->size(); position++) {
    reco::TrackRef tkTRef(tkTs,position);
    reco::TransientTrack tkTT(tkTRef);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;

    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();

    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    vm = theUpdator->update(outerTkTsos);
    TrajectoryStateOnSurface tkTsosFromTk = vm.stateAtTracker();

    std::pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);
    if (check.first) result.push_back(tkTRef);
  }

  return result;

}


//
//
//
std::vector<reco::TrackRef>
GlobalMuonTrackMatcher::match(const reco::TrackRef& staT, 
                              const std::vector<reco::TrackRef>& tkTs) const {

  std::vector<reco::TrackRef> result;

  reco::TransientTrack staTT(staT);

  TrajectoryStateOnSurface innerMuTsos = staTT.innermostMeasurementState();

// extrapolate innermost standalone TSOS to outer tracker surface
  MuonVertexMeasurement vm = theUpdator->update(innerMuTsos);
  TrajectoryStateOnSurface tkTsosFromMu = vm.stateAtTracker();

  for (std::vector<reco::TrackRef>::const_iterator tkTRef = tkTs.begin();
       tkTRef != tkTs.end(); tkTRef++) {
    reco::TransientTrack tkTT(*tkTRef);
    // make sure the tracker Track has enough momentum to reach muon chambers
    const GlobalVector& mom = tkTT.impactPointState().globalMomentum();
    if ( mom.mag() < theMinP || mom.perp() < theMinPt ) continue;

    TrajectoryStateOnSurface outerTkTsos = tkTT.outermostMeasurementState();

    // extrapolate outermost tracker measurement TSOS to outer tracker surface
    vm = theUpdator->update(outerTkTsos);
    TrajectoryStateOnSurface tkTsosFromTk = vm.stateAtTracker();

    std::pair<bool,double> check = match(tkTsosFromMu,tkTsosFromTk);

    if ( check.first ) result.push_back(*tkTRef);
  }

  return result;

}


//
// determine if two Tracks are compatible
// by comparing their TSOSs on the outer Tracker surface
//
std::pair<bool,double> 
GlobalMuonTrackMatcher::match(const reco::Track& sta, 
                              const reco::Track& tk) const {

  reco::TransientTrack staT(sta);  
  reco::TransientTrack tkT(tk);

  // make sure the tracker Track has enough momentum to reach muon chambers
  const GlobalVector& mom = tkT.impactPointState().globalMomentum();
   if ( mom.mag() < theMinP || mom.perp() < theMinPt )
     return std::pair<bool,double>(false,0);

  TrajectoryStateOnSurface outerTkTsos = tkT.outermostMeasurementState();
  TrajectoryStateOnSurface innerMuTsos = staT.innermostMeasurementState();

  // extrapolate innermost standalone TSOS to outer tracker surface
  MuonVertexMeasurement vm = theUpdator->update(innerMuTsos);
  TrajectoryStateOnSurface tkTsosFromMu = vm.stateAtTracker();

  // extrapolate outermost tracker measurement TSOS to outer tracker surface
  vm = theUpdator->update(outerTkTsos);
  TrajectoryStateOnSurface tkTsosFromTk = vm.stateAtTracker();

  // compare the TSOSs on outer tracker surface
  return match(tkTsosFromMu,tkTsosFromTk);

}


//
// determine if two TSOSs are compatible, they should be on same surface,
//  the outer tracker surface
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

