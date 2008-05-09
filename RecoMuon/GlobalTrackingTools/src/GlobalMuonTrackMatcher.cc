/**
 *  Class: GlobalMuonTrackMatcher
 *
 * 
 *  $Date: 2008/05/09 20:20:38 $
 *  $Revision: 1.11 $
 *
 *  \author Chang Liu - Purdue University
 *  \author Norbert Neumeister - Purdue University
 *  \author Adam Everett - Purdue University
 *
 */

#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

//---------------
// C++ Headers --
//---------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

using namespace std;
using namespace reco;

//
// constructor
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(const edm::ParameterSet& par, 
                                               const MuonServiceProxy* service) : 
   theService(service) {
  
  theMaxChi2 = par.getParameter<double>("Chi2Cut");
  theDeltaD = par.getParameter<double>("DeltaDCut");
  theDeltaR = par.getParameter<double>("DeltaRCut");
  theMinP = par.getParameter<double>("MinP");
  theMinPt = par.getParameter<double>("MinPt");
 
  theOutPropagatorName = par.getParameter<string>("Propagator");

}


//
// destructor
//
GlobalMuonTrackMatcher::~GlobalMuonTrackMatcher() {

}


//
// check if two tracks are compatible
//
bool 
GlobalMuonTrackMatcher::match(const TrackCand& sta,
                              const TrackCand& track) const {

  std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair
      = convertToTSOSMuHit(sta,track);

  double chi2 = match_Chi2(tsosPair.first,tsosPair.second);
  if ( chi2 > 0. && chi2 < theMaxChi2 ) return true;

  double distance = match_D(tsosPair.first,tsosPair.second);
  if ( distance > 0. && distance < theDeltaD ) return true;

  double deltaR = match_Rpos(tsosPair.first,tsosPair.second);
  if ( deltaR > 0. && deltaR < theDeltaR ) return true;

  return false;

}


//
// check if two tracks are compatible
//
double
GlobalMuonTrackMatcher::match(const TrackCand& sta,
                              const TrackCand& track,
                              int matchOption,
                              int surfaceOption) const {

  std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair;
  if ( surfaceOption == 0 ) tsosPair = convertToTSOSMuHit(sta,track);
  if ( surfaceOption == 1 ) tsosPair = convertToTSOSTkHit(sta,track);
  if ( surfaceOption != 0 && surfaceOption != 1 ) return -1.0;

  if ( matchOption == 0 ) {
    // chi^2
    return  match_Chi2(tsosPair.first,tsosPair.second);
  }
  else if ( matchOption == 1 ) {
    // distance
    return match_d(tsosPair.first,tsosPair.second);
  }
  else if ( matchOption == 2 ) {
    // deltaR
    return match_Rpos(tsosPair.first,tsosPair.second);
  }
  else {
    return -1.0;
  }

}


//
// choose the track from a TrackCollection which best
// matches a given standalone muon track
//
std::vector<GlobalMuonTrackMatcher::TrackCand>::const_iterator 
GlobalMuonTrackMatcher::matchOne(const TrackCand& sta,
                                 const std::vector<TrackCand>& tracks) const {

  if ( tracks.empty() ) return tracks.end();

  double minChi2 = 1000.0;
  vector<TrackCand>::const_iterator result = tracks.end();
  for (vector<TrackCand>::const_iterator is = tracks.begin(); is != tracks.end(); ++is) {
    // propagate to common surface
    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair
      = convertToTSOSMuHit(sta,*is);

    // calculate chi^2 of local track parameters
    double chi2 = match_Chi2(tsosPair.first,tsosPair.second);
    if ( chi2 <= minChi2 ) {
      minChi2 = chi2;
      result = is;
    }

  }

  return result;

}


//
// choose a vector of tracks from a TrackCollection that are compatible
// with a given standalone track. The order of checks for compatability are
// * matching-chi2 less than MaxChi2
// * gloabl position of TSOS on tracker bound
// * global momentum direction
//
vector<GlobalMuonTrackMatcher::TrackCand>
GlobalMuonTrackMatcher::match(const TrackCand& sta, 
                              const vector<TrackCand>& tracks) const {
  
  const string category = "GlobalMuonTrackMatcher";

  vector<TrackCand> result;
  
  if ( tracks.empty() ) return result;

  typedef std::pair<TrackCand, TrajectoryStateOnSurface> TrackCandWithTSOS;
  vector<TrackCandWithTSOS> cands;

  TrajectoryStateOnSurface muonTSOS;

  for (vector<TrackCand>::const_iterator is = tracks.begin(); is != tracks.end(); ++is) {

    // propagate to common surface
    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair
      = convertToTSOSMuHit(sta,*is);

    muonTSOS = tsosPair.first;
    cands.push_back(TrackCandWithTSOS(*is,tsosPair.second));
  }


  // try various matching criteria
  for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is) {

    double chi2 = match_Chi2(muonTSOS,(*is).second);
 
    if ( chi2 > 0. && chi2 < theMaxChi2 ) {
      result.push_back((*is).first);
    }
  }
 
  if ( result.empty() ) {
    LogDebug(category) << "MatchChi2 returned 0 results";
    for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is) {

      double distance = match_D(muonTSOS,(*is).second);

      if ( distance > 0. && distance < theDeltaD ) {
	result.push_back((*is).first);
      }
    }
  }
  
  if ( result.empty() ) {
    LogDebug(category) << "MatchD returned 0 results";
    for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is) {

      double deltaR = match_Rpos(muonTSOS,(*is).second);

      if ( deltaR > 0. && deltaR < theDeltaR ) {
        result.push_back((*is).first);
      }
    }
  }
 
  if ( result.empty() ) {
    LogDebug(category) << "MatchPos returned 0 results";
    TrackCand returnVal;
    double dR = 10.0;
    for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is) {
      double tmpR = match_R_IP(sta,(*is).first);
      if (tmpR < dR) {
	dR = tmpR;
	returnVal = (*is).first;
      }
    }
    result.push_back(returnVal);
  }

  return result;
 
}


//
// propagate the two track candidates to the tracker bound surface
//
std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSTk(const TrackCand& staCand,
                                        const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";

  TrajectoryStateOnSurface empty; 
  
  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface impactMuTSOS = muTT.impactPointState();

  TrajectoryStateOnSurface outerTkTsos;
  if (tkCand.second.isNonnull()) {
    // make sure the tracker track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->firstMeasurement().updatedState().globalMomentum();
    if ( !(mom.mag() < theMinP || mom.perp() < theMinPt)) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }
  
  if ( !impactMuTSOS.isValid() || !outerTkTsos.isValid() ) return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  
  // define StateOnTrackerBound object
  StateOnTrackerBound fromInside(&*theService->propagator(theOutPropagatorName));
  
  // extrapolate to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = fromInside(impactMuTSOS);
  TrajectoryStateOnSurface tkTsosFromTk = fromInside(outerTkTsos);

  if ( !samePlane(tkTsosFromMu,tkTsosFromTk) ) {
    // propagate tracker track to same surface as muon
    bool same1, same2;
    TrajectoryStateOnSurface newTkTsosFromTk, newTkTsosFromMu;
    if ( tkTsosFromMu.isValid() ) newTkTsosFromTk = theService->propagator(theOutPropagatorName)->propagate(outerTkTsos,tkTsosFromMu.surface());
    same1 = samePlane(newTkTsosFromTk,tkTsosFromMu);
    LogDebug(category) << "Propagating to same tracker surface (Mu):" << same1;
    if ( !same1 ) {
      if ( tkTsosFromTk.isValid() ) newTkTsosFromMu = theService->propagator(theOutPropagatorName)->propagate(impactMuTSOS,tkTsosFromTk.surface());
      same2 = samePlane(newTkTsosFromMu,tkTsosFromTk);
      LogDebug(category) << "Propagating to same tracker surface (Tk):" << same2;
    }
    if (same1) tkTsosFromTk = newTkTsosFromTk;
    else if (same2) tkTsosFromMu = newTkTsosFromMu;
    else  {
      LogDebug(category) << "Could not propagate Muon and Tracker track to the same tracker bound!";
      return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty, empty);
    }
  }
 
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(tkTsosFromMu, tkTsosFromTk);

}


//
// propagate the two track candidates to the surface of the innermost muon hit
//
std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSMuHit(const TrackCand& staCand,
                                           const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";

  TrajectoryStateOnSurface empty; 

  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();

  TrajectoryStateOnSurface outerTkTsos;
  if ( tkCand.second.isNonnull() ) {
    // make sure the tracker track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt ) ) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->lastMeasurement().updatedState().globalMomentum();
    if  ( !(mom.mag() < theMinP || mom.perp() < theMinPt) ) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }

  if ( !innerMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogDebug(category) << "A TSOS validity problem! MuTSOS " << innerMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }
  
  const Surface& refSurface = innerMuTSOS.surface();
  TrajectoryStateOnSurface tkAtMu = theService->propagator(theOutPropagatorName)->propagate(*outerTkTsos.freeState(),refSurface);
  
  if ( !tkAtMu.isValid() ) {
    LogDebug(category) << "Could not propagate Muon and Tracker track to the same muon hit surface!";
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);    
  }
  
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(innerMuTSOS, tkAtMu);

}


//
// propagate the two track candidates to the surface of the outermost tracker hit
//
std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSTkHit(const TrackCand& staCand,
                                           const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";

  TrajectoryStateOnSurface empty; 

  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface impactMuTSOS = muTT.impactPointState();

  TrajectoryStateOnSurface outerTkTsos;
  if ( tkCand.second.isNonnull() ) {
    // make sure the tracker track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->lastMeasurement().updatedState().globalMomentum();
    if (!(mom.mag() < theMinP || mom.perp() < theMinPt)) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }

  if ( !impactMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogDebug(category) << "A TSOS validity problem! MuTSOS " << impactMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }

  const Surface& refSurface = outerTkTsos.surface();
  TrajectoryStateOnSurface muAtTk = theService->propagator(theOutPropagatorName)->propagate(*impactMuTSOS.freeState(),refSurface);
  
  if ( !muAtTk.isValid() ) {
    LogDebug(category) << "Could not propagate Muon and Tracker track to the same tracker hit surface!";
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);    
  }

  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(muAtTk, outerTkTsos);

}


//
//
//
bool
GlobalMuonTrackMatcher::samePlane(const TrajectoryStateOnSurface& tsos1,
				  const TrajectoryStateOnSurface& tsos2) const {

  if ( !tsos1.isValid() || !tsos2.isValid() ) return false;

  if ( abs(match_D(tsos1,tsos2) - match_d(tsos1,tsos2)) > 0.1 ) return false;

  const float maxtilt = 0.999;
  const float maxdist = 0.01; // in cm

  ReferenceCountingPointer<TangentPlane> p1(tsos1.surface().tangentPlane(tsos1.localPosition()));
  ReferenceCountingPointer<TangentPlane> p2(tsos2.surface().tangentPlane(tsos2.localPosition()));

  bool returnValue = ( (fabs(p1->normalVector().dot(p2->normalVector())) > maxtilt) || (fabs((p1->toLocal(p2->position())).z()) < maxdist) ) ? true : false;

  return returnValue; 
 
}


//
// calculate Chi^2 of two trajectory states
//
double 
GlobalMuonTrackMatcher::match_Chi2(const TrajectoryStateOnSurface& tsos1, 
                                   const TrajectoryStateOnSurface& tsos2) const {
  
  const string category = "GlobalMuonTrackMatcher";
  
  if ( !tsos1.isValid() || !tsos2.isValid() ) return -1.;
  
  AlgebraicVector5 v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix55 m(tsos1.localError().matrix() + tsos2.localError().matrix());
  
  LogDebug(category) << "vector v " << v;

  bool ierr = !m.Invert();
 
  if ( !ierr ) edm::LogInfo(category) << "Error inverting covariance matrix";
 
  double est = ROOT::Math::Similarity(v,m);
 
  LogDebug(category) << "Chi2 " << est;

  return est;

}


//
// calculate Delta_R of two track candidates at the IP
//
double
GlobalMuonTrackMatcher::match_R_IP(const TrackCand& staCand, 
                                   const TrackCand& tkCand) const {

  double dR = 99.0;
  if (tkCand.second.isNonnull()) {
    dR = (deltaR<double>(staCand.second->eta(),staCand.second->phi(),
			 tkCand.second->eta(),tkCand.second->phi()));
  } else {
    dR = (deltaR<double>(staCand.second->eta(),staCand.second->phi(),
			 tkCand.first->firstMeasurement().updatedState().globalMomentum().eta(),
			 tkCand.first->firstMeasurement().updatedState().globalMomentum().phi()));
  }

  return dR;

}


//
// calculate Delta_R of two trajectory states
//
double
GlobalMuonTrackMatcher::match_Rmom(const TrajectoryStateOnSurface& sta, 
                                   const TrajectoryStateOnSurface& tk) const {

  if( !sta.isValid() || !tk.isValid() ) return -1;
  return (deltaR<double>(sta.globalMomentum().eta(),sta.globalMomentum().phi(),
			 tk.globalMomentum().eta(),tk.globalMomentum().phi()));

}


//
// calculate Delta_R of two trajectory states
//
double
GlobalMuonTrackMatcher::match_Rpos(const TrajectoryStateOnSurface& sta, 
                                   const TrajectoryStateOnSurface& tk) const {

  if ( !sta.isValid() || !tk.isValid() ) return -1;
  return (deltaR<double>(sta.globalPosition().eta(),sta.globalPosition().phi(),
			 tk.globalPosition().eta(),tk.globalPosition().phi()));

}


//
// calculate the distance in global position of two trajectory states
//
double
GlobalMuonTrackMatcher::match_D(const TrajectoryStateOnSurface& sta, 
                                const TrajectoryStateOnSurface& tk) const {

  if ( !sta.isValid() || !tk.isValid() ) return -1;
  return (sta.globalPosition() - tk.globalPosition()).mag();

}


//
// calculate the distance in local position of two trajectory states
//
double
GlobalMuonTrackMatcher::match_d(const TrajectoryStateOnSurface& sta, 
                                const TrajectoryStateOnSurface& tk) const {

  if ( !sta.isValid() || !tk.isValid() ) return -1;
  return (sta.localPosition() - tk.localPosition()).mag();

}

double
GlobalMuonTrackMatcher::match_dist(const TrajectoryStateOnSurface& sta, 
				   const TrajectoryStateOnSurface& tk) const {
  
  const string category = "GlobalMuonTrackMatcher";
  
  if ( !sta.isValid() || !tk.isValid() ) return -1;
  
  AlgebraicMatrix22 M;
  M(0,0) =  tk.localError().positionError().xx()+ sta.localError().positionError().xx();
  M(1,0) = M(0,1) = tk.localError().positionError().xy()+ sta.localError().positionError().xy();
  M(1,1) =  tk.localError().positionError().yy()+ sta.localError().positionError().yy();
  AlgebraicVector2 v;
  v[0] = tk.localDirection().x()- sta.localDirection().x();
  v[1] = tk.localDirection().y()- sta.localDirection().y();
  
  if(!M.Invert()){
    LogDebug(category) << "Error inverting local matrix ";
    return -1;
  }
  
  return  ROOT::Math::Similarity(v,M);
}
