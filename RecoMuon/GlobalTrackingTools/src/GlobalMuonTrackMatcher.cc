/**
 *  Class: GlobalMuonTrackMatcher
 *
 * 
 *  $Date: $
 *  $Revision: $
 *
 *  Authors :
 *  \author Chang Liu  - Purdue University
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
#include "Utilities/Timing/interface/TimingReport.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

using namespace std;
using namespace reco;

//
// constructor
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(const edm::ParameterSet& par, 
                                               const MuonServiceProxy* service) : 
   theService(service) {
  
  theMaxChi2 =  par.getParameter<double>("Chi2Cut");
  theDeltaEta = par.getParameter<double>("DeltaEtaCut");
  theDeltaPhi = par.getParameter<double>("DeltaPhiCut");
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
  
  bool hasMatchTk = false;
  TrackCand result = staCand;
  double minChi2 = theMaxChi2;
  
  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {

    pair<bool,double> check = matchChi(staCand,*is);
    
    if (!check.first) continue;
    
    if (check.second < minChi2) { 
      hasMatchTk = true;
      minChi2 = check.second;
      result = (*is);
    } 
  }     

  return pair<bool, TrackCand>(hasMatchTk, result);

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
    pair<bool,double> check = matchChi(staCand,*is);    
    if ( check.first ) result.push_back(*is);
  }

  if ( result.empty() ) {
    LogDebug(category) << "MatchChi returned 0 results";
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      if ( matchPos(staCand,*is) ) result.push_back(*is);
    }
  }

  // if there are no matches, return the TkTrack closest to STACand in eta-phi space
  if ( result.empty() ) {
    LogDebug(category) << "MatchPos returned 0 results";
    result.push_back(matchMomAtIP(staCand,tkTs));
  }

  return result;

}


/*!  
  \return pair<match result, chi2>
  \see matchChiAtSurface()
 */
pair<bool,double> 
GlobalMuonTrackMatcher::matchChi(const TrackCand& staCand, 
                                 const TrackCand& tkCand) const {

  const string category = "GlobalMuonTrackMatcher";
  double chi2 = -1;
  
  pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair =   
    convertToTSOS(staCand,tkCand);
  
  bool sameSurface = samePlane(tsosPair.first,tsosPair.second);
  
  LogDebug(category) << "Match at surface";
  if( sameSurface ) 
    chi2 = matchChiAtSurface(tsosPair.first, tsosPair.second);
  
  bool chi2Match = ( chi2 > 0. && chi2 <= theMaxChi2 ) ? true : false;
  return pair<bool,double>(chi2Match,chi2);
  
}


/*!
 * Compare the global position of two track candidates on tracker bound
 * \return true if gloabl positions are less than DeltaEta or DeltaPhi
 */
bool
GlobalMuonTrackMatcher::matchPos(const TrackCand& staCand,
				 const TrackCand& tkCand) const {

  const string category = "GlobalMuonTrackMatcher";

  pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = 
      convertToTSOS(staCand,tkCand);

  return matchPosAtSurface(tsosPair.first, tsosPair.second);

}


/*!
 * Take two TrackCands and calculate their TSOSs on the outer tracker
 * surface.
 */
pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOS(const TrackCand& staCand,
				      const TrackCand& tkCand) const {
  
  const string category = "GlobalMuonTrackMatcher";

  TransientTrack muTT(*staCand.second,&*theService->magneticField(),theService->trackingGeometry());
  TrajectoryStateOnSurface innerMuTSOS = muTT.impactPointState();
  FreeTrajectoryState initMuFTS = muTT.initialFreeState();

  TrajectoryStateOnSurface outerTkTsos;
  if (tkCand.first == 0) {
    LogDebug(category);
    // make sure the tracker Track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      TrajectoryStateTransform tsTransform;
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {    
    LogDebug(category);
    const GlobalVector& mom = tkCand.first->firstMeasurement().updatedState().globalMomentum();
    if ( ! (mom.mag() < theMinP || mom.perp() < theMinPt )) {
      outerTkTsos = (tkCand.first->direction() == alongMomentum) ? tkCand.first->lastMeasurement().updatedState() : tkCand.first->firstMeasurement().updatedState();
    }
  }

  if ( !innerMuTSOS.isValid() || !outerTkTsos.isValid() ) return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(innerMuTSOS,outerTkTsos);

  // define StateOnTrackerBound objects  
  StateOnTrackerBound fromInside(&*theService->propagator(theOutPropagatorName));

  // extrapolate to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = fromInside(initMuFTS);
  TrajectoryStateOnSurface tkTsosFromTk = fromInside(outerTkTsos);

    
  if( !samePlane(tkTsosFromMu,tkTsosFromTk)) {
    bool same1, same2;
    //propagate tk to same surface as muon
    TrajectoryStateOnSurface newTkTsosFromTk, newTkTsosFromMu;
    if( tkTsosFromMu.isValid() ) newTkTsosFromTk = theService->propagator(theOutPropagatorName)->propagate(outerTkTsos,tkTsosFromMu.surface());
    same1 =  samePlane(newTkTsosFromTk,tkTsosFromMu);
    LogDebug(category) << "Propagating to same surface (Mu):" << same1;
    if( !same1 ) {
      if( tkTsosFromTk.isValid() ) newTkTsosFromMu = theService->propagator(theOutPropagatorName)->propagate(initMuFTS,tkTsosFromTk.surface());
      same2 =  samePlane(newTkTsosFromMu,tkTsosFromTk);
      LogDebug(category) << "Propagating to same surface (Tk):" << same2;
    }
    if(same1) tkTsosFromTk = newTkTsosFromTk;
    else if(same2) tkTsosFromMu = newTkTsosFromMu;
    else  LogDebug(category) << "Could not propagate Muon and Tracker track to the same tracker bound!";
  }
  

  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(tkTsosFromMu, tkTsosFromTk);

}


/*!
 * Determine if two TSOSs are compatible; they should be on same surface.
 * \return chi2
 */ 
double 
GlobalMuonTrackMatcher::matchChiAtSurface(const TrajectoryStateOnSurface& tsos1, 
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

/*
  GlobalVector x = tsos1.globalParameters().position() - tsos2.globalParameters().position();
  AlgebraicVector v1(3); v1[0] = x.x(); v1[1] = x.y(); v1[2] = x.z();
  AlgebraicSymMatrix m1(tsos1.cartesianError().position().matrix() + tsos2.cartesianError().position().matrix());
  m1.invert(ierr);
  double est1 = m1.similarity(v1);
*/

  return est;

}



/*!
 * Compare global eta-phi position of two TSOSs.
 */
bool
GlobalMuonTrackMatcher::matchPosAtSurface(const TrajectoryStateOnSurface& tsos1,
                                          const TrajectoryStateOnSurface& tsos2) const {

  const string category = "GlobalMuonTrackMatcher";

  if ( !tsos1.isValid() || !tsos2.isValid() ) return false;

  double phi1 = tsos1.globalPosition().phi();
  double phi2 = tsos2.globalPosition().phi();
  double dphi(fabs(Geom::Phi<double>(phi1)-Geom::Phi<double>(phi2)));

  double eta1 = tsos1.globalPosition().eta();
  double eta2 = tsos2.globalPosition().eta();
  double deta(fabs(eta1-eta2));

  bool good = ( (dphi < theDeltaPhi) || (deta < theDeltaEta) ) ? true : false;
  LogDebug(category) << "dphi " << dphi << " deta " << deta;

  return good;

}


/*!  Find the one TrackCand in a collection of TrackCands with the global
  direction closest to the given standalone muon.  
  \param staCand given strandalone muon.
  \return TrackCand with momentum direction closest to that of staCand
 */
GlobalMuonTrackMatcher::TrackCand
GlobalMuonTrackMatcher::matchMomAtIP(const TrackCand& staCand,
                                     const std::vector<TrackCand>& tkTs) const {

  const string category = "GlobalMuonTrackMatcher";

  TrackCand returnVal;
  float deltaR = 1000.0;
  
  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    double eta1 = staCand.second->eta();
    double eta2;
    if ((*is).first != 0) {
      eta2 = (*is).first->firstMeasurement().updatedState().globalMomentum().eta();
    } else {
      eta2 = (*is).second->eta();
    }
    double phi1 = staCand.second->phi();
    double phi2;
    if ((*is).first != 0) {
      phi2 = (*is).first->firstMeasurement().updatedState().globalMomentum().phi();
    } else {
      phi2 = (*is).second->phi();
    }
    double deltaEta = eta1 - eta2;
    double deltaPhi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));
    double deltaR_tmp = sqrt(deltaEta*deltaEta + deltaPhi*deltaPhi);

    if (deltaR_tmp < deltaR) {
      deltaR = deltaR_tmp;
      returnVal = *is;
    }
  }    

  return returnVal;  

}

/*!
 * In the case that the TSOS is on a cylinder, check the TSOS' TangentialPlane.
 */
bool GlobalMuonTrackMatcher::samePlane(const TrajectoryStateOnSurface& tsos1,
				       const TrajectoryStateOnSurface& tsos2) const
{
  if( !tsos1.isValid() || !tsos2.isValid()) return false;
  const string category = "GlobalMuonTrackMatcher";

  const float maxtilt = 0.999;
  const float maxdist = 0.01; // in cm

  ReferenceCountingPointer<TangentPlane> p1(tsos1.surface().tangentPlane(tsos1.localPosition()));
  ReferenceCountingPointer<TangentPlane> p2(tsos2.surface().tangentPlane(tsos2.localPosition()));

  bool returnValue =  ( (fabs(p1->normalVector().dot(p2->normalVector())) > maxtilt) || (fabs((p1->toLocal(p2->position())).z()) < maxdist) ) ? true : false;

  return returnValue; 
  
}
