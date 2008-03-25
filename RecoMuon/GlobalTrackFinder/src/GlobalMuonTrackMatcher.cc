/**
 *  Class: GlobalMuonTrackMatcher
 *
 * 
 *  $Date: 2008/02/29 15:54:08 $
 *  $Revision: 1.45.4.2 $
 *
 *  Authors :
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 *  \author Adam Everett - Purdue University
 *
 */

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"

//---------------
// C++ Headers --
//---------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
//#include "TrackingTools/GeomPropagators/interface/StateOnMuonBound.h"
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
  
  theMaxChi2 =  par.getParameter<double>("Chi2Cut");
  theDeltaD = par.getParameter<double>("DeltaDCut");
  theDeltaR = par.getParameter<double>("DeltaRCut");
  theMinP = par.getParameter<double>("MinP");
  theMinPt = par.getParameter<double>("MinPt");
  
  theOutPropagatorName = par.getParameter<string>("StateOnTrackerBoundOutPropagator");

}


//
// destructor
//
GlobalMuonTrackMatcher::~GlobalMuonTrackMatcher() {

}


/*!
 * Choose the Track from a TrackCollection which has smallest chi2 with
 * a given standalone muon Track.
 */
pair<bool, GlobalMuonTrackMatcher::TrackCand> 
GlobalMuonTrackMatcher::matchOne(const TrackCand& staCand,
				 const vector<TrackCand>& tkTs) const {

  return pair<bool, TrackCand>(false, staCand);
  
}


/*!
 * Choose a vector of Tracks from a TrackCollection that are compatible
 * with a given standalone Track.  The order of checks for compatability are
 * \li matching-chi2 less than MaxChi2
 * \li gloabl position of TSOS on tracker bound
 * \li global momentum direction
 * \see matchChi()
 * \see matchPos()
 * \see matchMomAtIP()
 */
vector<GlobalMuonTrackMatcher::TrackCand>
GlobalMuonTrackMatcher::match(const TrackCand& staCand, 
                              const vector<TrackCand>& tkTs) const {
  
  const string category = "GlobalMuonTrackMatcher";  
  vector<TrackCand> result; 
  
  if ( tkTs.empty() ) return result;

  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {    

    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPairMuHit
      = convertToTSOSMuHit(staCand,*is);

    double fillVal = match_ChiAtSurface(tsosPairMuHit.first,tsosPairMuHit.second);
    
    if( fillVal > 0. && fillVal < theMaxChi2 ) {
      result.push_back(*is);
    } else {
    }
  }
  
  if(result.empty()) {
    LogDebug(category) << "MatchChi returned 0 results";
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPairMuHit
	= convertToTSOSMuHit(staCand,*is);

      double fillVal = match_D(tsosPairMuHit.first,tsosPairMuHit.second);

      if( fillVal > 0. && fillVal < theDeltaD ) {
	result.push_back(*is);        
      }
    }
  }
  
  if(result.empty()) {
    LogDebug(category) << "MatchD returned 0 results";
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPairMuHit
	= convertToTSOSMuHit(staCand,*is);

      double fillVal = match_Rpos(tsosPairMuHit.first,tsosPairMuHit.second);

      if( fillVal > 0. && fillVal < theDeltaR ) result.push_back(*is); 
    }
  }
 
  if(result.empty()) {
    LogDebug(category) << "MatchPos returned 0 results";
    TrackCand returnVal;
    double dR = 10.0;
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      double tmpR = match_R_IP(staCand,*is);
      
      if (tmpR < dR) {
	dR = tmpR;
	returnVal = *is;
      }
    }

    result.push_back(returnVal);
  }

  
  return result;
  
}



std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSTk(const TrackCand& staCand,
			    const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";
  TrajectoryStateOnSurface empty; 
  
  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface impactMuTSOS = muTT.impactPointState();

  TrajectoryStateOnSurface outerTkTsos;
  if( tkCand.second.isNonnull() ) {
    //make sure the trackerTrack has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->firstMeasurement().updatedState().globalMomentum();
    if(!(mom.mag() < theMinP || mom.perp() < theMinPt)) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }
  
  if ( !impactMuTSOS.isValid() || !outerTkTsos.isValid() ) return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  
  // define StateOnTrackerBound objects  
  StateOnTrackerBound fromInside(&*theService->propagator(theOutPropagatorName));
  
  // extrapolate to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = fromInside(impactMuTSOS);
  TrajectoryStateOnSurface tkTsosFromTk = fromInside(outerTkTsos);

    
  if( !samePlane(tkTsosFromMu,tkTsosFromTk)) {
    bool same1, same2;
    //propagate tk to same surface as muon
    TrajectoryStateOnSurface newTkTsosFromTk, newTkTsosFromMu;
    if( tkTsosFromMu.isValid() ) newTkTsosFromTk = theService->propagator(theOutPropagatorName)->propagate(outerTkTsos,tkTsosFromMu.surface());
    same1 =  samePlane(newTkTsosFromTk,tkTsosFromMu);
    LogDebug(category) << "Propagating to same tracker surface (Mu):" << same1;
    if( !same1 ) {
      if( tkTsosFromTk.isValid() ) newTkTsosFromMu = theService->propagator(theOutPropagatorName)->propagate(impactMuTSOS,tkTsosFromTk.surface());
      same2 =  samePlane(newTkTsosFromMu,tkTsosFromTk);
      LogDebug(category) << "Propagating to same tracker surface (Tk):" << same2;
    }
    if(same1) tkTsosFromTk = newTkTsosFromTk;
    else if(same2) tkTsosFromMu = newTkTsosFromMu;
    else  {
      LogDebug(category) << "Could not propagate Muon and Tracker track to the same tracker bound!";
      return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty, empty);
    }
  }
  
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(tkTsosFromMu, tkTsosFromTk);
}


std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSMuHit(const TrackCand& staCand,
			       const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";
  TrajectoryStateOnSurface empty; 

  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();
  
  TrajectoryStateOnSurface outerTkTsos;
  if(tkCand.second.isNonnull()) {
    //make sure the trackerTrack has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->lastMeasurement().updatedState().globalMomentum();
    if(!(mom.mag() < theMinP || mom.perp() < theMinPt)) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }
  
  if ( !innerMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogDebug(category) << "A TSOS Validity problem! MuTSOS " << innerMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }
  
  const Surface & refSurface = innerMuTSOS.surface();
  
  TrajectoryStateOnSurface tkAtMu = theService->propagator(theOutPropagatorName)->propagate(*outerTkTsos.freeState(),refSurface);
    
  if(!tkAtMu.isValid()) {
    LogDebug(category) << "Could not propagate Muon and Tracker track to the same muon hit surface!";
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);    
  }  
  
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(innerMuTSOS, tkAtMu);
  
}


std::pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOSTkHit(const TrackCand& staCand,
			       const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";
  TrajectoryStateOnSurface empty; 

  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface impactMuTSOS = muTT.impactPointState();

  TrajectoryStateOnSurface outerTkTsos;
  if(tkCand.second.isNonnull()) {
    //make sure the trackerTrack has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {
    const GlobalVector& mom = tkCand.first->lastMeasurement().updatedState().globalMomentum();
    if(!(mom.mag() < theMinP || mom.perp() < theMinPt)) {
      outerTkTsos = tkCand.first->lastMeasurement().updatedState();
    }
  }

  if ( !impactMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogDebug(category) << "A TSOS Validity problem! MuTSOS " << impactMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }
  
  const Surface & refSurface = outerTkTsos.surface();
  
  TrajectoryStateOnSurface muAtTk = theService->propagator(theOutPropagatorName)->propagate(*impactMuTSOS.freeState(),refSurface);
  
  
  if( !muAtTk.isValid() ) {
    LogDebug(category) << "Could not propagate Muon and Tracker track to the same tracker hit surface!";
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);    
  }
  
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(muAtTk, outerTkTsos);

}


bool
GlobalMuonTrackMatcher::samePlane(const TrajectoryStateOnSurface& tsos1,
				  const TrajectoryStateOnSurface& tsos2) const
{
  if( !tsos1.isValid() || !tsos2.isValid()) return false;
  const string category = "GlobalMuonTrackMatcher";

  if(abs(match_D(tsos1,tsos2)-match_d(tsos1,tsos2))>0.1) return false;

  const float maxtilt = 0.999;
  const float maxdist = 0.01; // in cm

  ReferenceCountingPointer<TangentPlane> p1(tsos1.surface().tangentPlane(tsos1.localPosition()));
  ReferenceCountingPointer<TangentPlane> p2(tsos2.surface().tangentPlane(tsos2.localPosition()));

  bool returnValue =  ( (fabs(p1->normalVector().dot(p2->normalVector())) > maxtilt) || (fabs((p1->toLocal(p2->position())).z()) < maxdist) ) ? true : false;

  return returnValue; 
  
}

double 
GlobalMuonTrackMatcher::match_ChiAtSurface(const TrajectoryStateOnSurface& tsos1, 
			      const TrajectoryStateOnSurface& tsos2) const {
  
  const string category = "GlobalMuonTrackMatcher";
  
  if ( !tsos1.isValid() || !tsos2.isValid() ) return -1.;
  
  AlgebraicVector5 v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix55 m(tsos1.localError().matrix() + tsos2.localError().matrix());
  
  LogDebug(category) << "vector v " << v;

  int ierr = ! m.Invert();
  
  if (ierr != 0) edm::LogInfo(category) << "Error inversing covariance matrix";
  
  double est = ROOT::Math::Similarity(v,m);
  
  LogDebug(category) << "Chi2 " << est;

  return est;

}

double
GlobalMuonTrackMatcher::match_R_IP(const TrackCand& staCand, const TrackCand& tkCand) const {
  double dR = 99.0;
  if(tkCand.second.isNonnull()) {
    dR = (deltaR<double>(staCand.second->eta(),staCand.second->phi(),
			 tkCand.second->eta(),tkCand.second->phi()));
  } else {
    dR = (deltaR<double>(staCand.second->eta(),staCand.second->phi(),
			 tkCand.first->firstMeasurement().updatedState().globalMomentum().eta(),
			 tkCand.first->firstMeasurement().updatedState().globalMomentum().phi()));
  }
  
  return dR;
}


double
GlobalMuonTrackMatcher::match_Rmom(const TrajectoryStateOnSurface& sta, const TrajectoryStateOnSurface& tk) const {
  if( !sta.isValid() || !tk.isValid() ) return -1;
  return (deltaR<double>(sta.globalMomentum().eta(),sta.globalMomentum().phi(),
			 tk.globalMomentum().eta(),tk.globalMomentum().phi()));
}

double
GlobalMuonTrackMatcher::match_Rpos(const TrajectoryStateOnSurface& sta, const TrajectoryStateOnSurface& tk) const {
  if( !sta.isValid() || !tk.isValid() ) return -1;
  return (deltaR<double>(sta.globalPosition().eta(),sta.globalPosition().phi(),
			 tk.globalPosition().eta(),tk.globalPosition().phi()));
}

double
GlobalMuonTrackMatcher::match_D(const TrajectoryStateOnSurface& sta, const TrajectoryStateOnSurface& tk) const {
  if( !sta.isValid() || !tk.isValid() ) return -1;
  return (sta.globalPosition() - tk.globalPosition()).mag();
}

double
GlobalMuonTrackMatcher::match_d(const TrajectoryStateOnSurface& sta, const TrajectoryStateOnSurface& tk) const {
  if( !sta.isValid() || !tk.isValid() ) return -1;
  return (sta.localPosition() - tk.localPosition()).mag();
}
