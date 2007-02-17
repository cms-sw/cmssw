/**
 *  Class: GlobalMuonTrackMatcher
 *
 *  Description:
 *    Match standalone muon track with tracker tracks
 *
 *  $Date: 2007/02/16 23:41:47 $
 *  $Revision: 1.40 $
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
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonMonitorInterface.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

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
  
  matchAtSurface_ = par.getUntrackedParameter<bool>("MatchAtSurface",true);

  theOutPropagatorName = par.getParameter<string>("StateOnTrackerBoundOutPropagator");

  theMIMFlag = par.getUntrackedParameter<bool>("performMuonIntegrityMonitor",false);
  if (theMIMFlag) {
    dataMonitor = edm::Service<GlobalMuonMonitorInterface>().operator->();
  }

}


//
// destructor
//
GlobalMuonTrackMatcher::~GlobalMuonTrackMatcher() {

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


//
// choose a vector of tracker Tracks from a TrackCollection that has Chi2 
// less than theMaxChi2, for a given standalone Track
//
vector<GlobalMuonTrackMatcher::TrackCand>
GlobalMuonTrackMatcher::match(const TrackCand& staCand, 
                              const vector<TrackCand>& tkTs) const {

  const string category = "GlobalMuonTrackMatcher";  
  vector<TrackCand> result; 
  
  if ( tkTs.empty() ) return result;

  if (theMIMFlag) {
    dataMonitor->book1D("matchMethod","Match type result of event",11,-0.5,10.5);
    dataMonitor->fill1("matchMethod",0);
  }

  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    pair<bool,double> check = matchChi(staCand,*is);    
    if ( check.first ) result.push_back(*is);
  }

  if (theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",1);  
    
  if ( result.empty() ) {
    LogDebug(category) << "MatchChi returned 0 results";
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      if ( matchPos(staCand,*is) ) result.push_back(*is);
    }
  }

  if (theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",2);  
    
  // if there are no matches, return the TkTrack closest to STACand in eta-phi space
  if ( result.empty() ) {
    LogDebug(category) << "MatchPos returned 0 results";
    result.push_back(matchMomAtIP(staCand,tkTs));
  }

  if (theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",3);  

  return result;

}


//
// determine if two TrackRefs are compatible
// by comparing their TSOSs on the outer Tracker surface
//
pair<bool,double> 
GlobalMuonTrackMatcher::matchChi(const TrackCand& staCand, 
                                 const TrackCand& tkCand) const {

  const string category = "GlobalMuonTrackMatcher";
  double chi2 = -1;
  
  if ( matchAtSurface_ ) {
    LogDebug(category) << "Match at surface";
    pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = 
      convertToTSOS(staCand,tkCand);
    chi2 = matchChiAtSurface(tsosPair.first, tsosPair.second);
  } else {    
    LogDebug(category) << "Match at IP" ;
    chi2 = matchChiAtIP(staCand, tkCand);    
  }
  
  bool chi2Match = ( chi2 > 0. && chi2 <= theMaxChi2 ) ? true : false;

  if (theMIMFlag) {
    dataMonitor->book1D("matchChi2_all","#chi^{2} of all tracks",500,0,1000);
    dataMonitor->fill1("matchChi2_all",chi2);
  }
  
  return pair<bool,double>(chi2Match,chi2);

}


//
// compare the global position of to track candidatas at a given surface
//
bool
GlobalMuonTrackMatcher::matchPos(const TrackCand& staCand,
				 const TrackCand& tkCand) const {

  const string category = "GlobalMuonTrackMatcher";

  pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = 
      convertToTSOS(staCand,tkCand);

  return matchPosAtSurface(tsosPair.first, tsosPair.second);

}


//
// take two TrackCands and calcultae their 
// TSOSs on the outer tracker surface
//
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

  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(tkTsosFromMu, tkTsosFromTk);

}


//
// determine if two TSOSs are compatible; they should be on same surface
// 
double 
GlobalMuonTrackMatcher::matchChiAtSurface(const TrajectoryStateOnSurface& tsos1, 
					  const TrajectoryStateOnSurface& tsos2) const {

  const string category = "GlobalMuonTrackMatcher";

  if ( !tsos1.isValid() || !tsos2.isValid() ) return -1.;

  AlgebraicVector v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix m(tsos1.localError().matrix() + tsos2.localError().matrix());
  LogDebug(category) << "vector v " << v;

  int ierr;
  m.invert(ierr);
  if (ierr != 0) edm::LogInfo(category) << "Error inversing covariance matrix";
  double est = m.similarity(v);
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


//
// check if two Tracks match at IP
//
double
GlobalMuonTrackMatcher::matchChiAtIP(const TrackCand& staCand, 
                                     const TrackCand& tkCand) const {

  const string category = "GlobalMuonTrackMatcher";  

  TrackBase::ParameterVector delta = staCand.second->parameters() - tkCand.second->parameters();
  LogDebug(category) << "Parameter Vector " << delta;
  TrackBase::CovarianceMatrix cov = staCand.second->covariance()+tkCand.second->covariance();

  cov.Invert();
  double chi2 = 0.;
  for (unsigned int i=0; i<TrackBase::dimension; i++) {
    for (unsigned int j=0; j<TrackBase::dimension; j++) {
      chi2 += delta[i]*cov(i,j)*delta[j];
    }
  }

  return chi2;  

}


//
// compare global position of two TSOSs
//
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


//
// compare global directions of track candidates
//
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
