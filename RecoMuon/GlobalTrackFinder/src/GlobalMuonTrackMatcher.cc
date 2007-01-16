/** \class GlobalMuonTrackMatcher
 *  match standalone muon track with tracker tracks
 *
 *  $Date: 2006/12/11 00:03:41 $
 *  $Revision: 1.34 $
 *  \author Chang Liu  - Purdue University
 *  \author Norbert Neumeister - Purdue University
 *  \author Adam Everett - Purdue University
 */

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonMonitorInterface.h"
#include "Utilities/Timing/interface/TimingReport.h"

#include "DataFormats/TrackReco/interface/Track.h"


using namespace std;
using namespace edm;
using namespace reco;
//
// constructor
//
GlobalMuonTrackMatcher::GlobalMuonTrackMatcher(const edm::ParameterSet& par, 
                                               const MuonServiceProxy* service) : 
   theService(service) {
  
  ParameterSet updatorPSet = par.getParameter<ParameterSet>("UpdatorParameters");
  theUpdator = new MuonUpdatorAtVertex(updatorPSet,theService);
  
  theMaxChi2 =  par.getParameter<double>("Chi2CutTrackMatcher");
  theMinP = 2.5;
  theMinPt = 1.0;

  theMIMFlag = par.getUntrackedParameter<bool>("performMuonIntegrityMonitor",false);
  if(theMIMFlag) {
    dataMonitor = edm::Service<GlobalMuonMonitorInterface>().operator->(); 
  }

  matchAtSurface_ = par.getUntrackedParameter<bool>("MatchAtSurface",true);

}


//
//
//
GlobalMuonTrackMatcher::~GlobalMuonTrackMatcher() {

  if (theUpdator) delete theUpdator;

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
                              const std::vector<TrackCand>& tkTs) const {
  
  vector<TrackCand> result; 
  
  if ( tkTs.empty() ) return result;

  if (theMIMFlag) {
    dataMonitor->book1D("matchMethod","Match type result of event",11,-0.5,10.5);
    dataMonitor->fill1("matchMethod",0);
  }
    
  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    pair<bool,double> check = matchChi(staCand,*is);    
    
    if ( check.first ) {
      result.push_back(*is);
    }
  }

  if(theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",1);  
    
  if( result.empty() ) {
    for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
      if( matchPos(staCand,*is) ) result.push_back(*is);
    }
  }

  if(theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",2);  
    
  //if there are no matches, return the TkTrack closest to STACandin eta-phi space
  if ( result.empty() ) {
    result.push_back(matchMomAtIP(staCand,tkTs));
  }

  if(theMIMFlag && !result.empty()) dataMonitor->fill1("matchMethod",3);  
  
  return result;
}


//
// determine if two TrackRefs are compatible
// by comparing their TSOSs on the outer Tracker surface
//
pair<bool,double> 
GlobalMuonTrackMatcher::matchChi(const TrackCand& staCand, 
                              const TrackCand& tkCand) const {

  double chi2 = -1;
  
  if( matchAtSurface_ ) {
    pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = 
      convertToTSOS(staCand,tkCand);    
    chi2 = matchChiAtSurface(tsosPair.first, tsosPair.second);        
  } else {    
    chi2 = matchChiAtIP(staCand, tkCand);    
  }
  
  bool chi2Match = ( chi2 > 0. && chi2 <= theMaxChi2 ) ? true : false;
    
  if(theMIMFlag) {
    dataMonitor->book1D("matchChi2_all","#chi^{2} of all tracks",500,0,1000);
    dataMonitor->fill1("matchChi2_all",chi2);
  }
  
  return pair<bool,double>(chi2Match,chi2);
}


bool
GlobalMuonTrackMatcher::matchPos(const TrackCand& staCand,
				 const TrackCand& tkCand) const {
  
  pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = 
      convertToTSOS(staCand,tkCand);
  
  return matchPosAtSurface(tsosPair.first, tsosPair.second);
}


pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>
GlobalMuonTrackMatcher::convertToTSOS(const TrackCand& staCand,
				      const TrackCand& tkCand) const {
  
  TrajectoryStateOnSurface innerMuTsos;  
  TrajectoryStateOnSurface outerTkTsos;
  TrajectoryStateTransform tsTransform;
  
  if (staCand.first == 0) {
    if(theMIMFlag) {
      dataMonitor->book1D("matchPropTime","Propagation time from innerMu to Tk Surface",1000,0.,4.);
    }
    TimeMe propTime("matchProp");
    innerMuTsos = tsTransform.innerStateOnSurface(*staCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    pair<double,double> time = propTime.lap();
    if(theMIMFlag) dataMonitor->fill1("matchPropTime",time.second);
  } else {
    innerMuTsos = staCand.first->firstMeasurement().updatedState();
  }
  
  if (tkCand.first == 0) {
    // make sure the tracker Track has enough momentum to reach muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      outerTkTsos = tsTransform.outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
    }
  } else {    
    const GlobalVector& mom = tkCand.first->firstMeasurement().updatedState().globalMomentum();
    if ( ! (mom.mag() < theMinP || mom.perp() < theMinPt )) {
      outerTkTsos = (tkCand.first->direction() == alongMomentum) ? tkCand.first->lastMeasurement().updatedState() : tkCand.first->firstMeasurement().updatedState();
    }
  }

  if( !innerMuTsos.isValid() || !outerTkTsos.isValid() ) return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(innerMuTsos,outerTkTsos);

  // extrapolate innermost standalone TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromMu = theUpdator->stateAtTracker(innerMuTsos);

  // extrapolate outermost tracker measurement TSOS to outer tracker surface
  TrajectoryStateOnSurface tkTsosFromTk = theUpdator->stateAtTracker(outerTkTsos);
  
  return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(tkTsosFromMu, tkTsosFromTk);
  
}


//
// determine if two TSOSs are compatible, they should be on same surface
// 
double 
GlobalMuonTrackMatcher::matchChiAtSurface(const TrajectoryStateOnSurface& tsos1, 
					  const TrajectoryStateOnSurface& tsos2) const {

  if( !tsos1.isValid() || !tsos2.isValid() ) return -1.;

  AlgebraicVector v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix m(tsos1.localError().matrix() + tsos2.localError().matrix());
  
  int ierr;
  m.invert(ierr);
  // if (ierr != 0) throw exception;
  double est = m.similarity(v);
  
  return est;
}


double
GlobalMuonTrackMatcher::matchChiAtIP(const TrackCand& staCand, 
				     const TrackCand& tkCand) const {
  
  TrackBase::ParameterVector delta = staCand.second->parameters() - tkCand.second->parameters();
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


bool
GlobalMuonTrackMatcher::matchPosAtSurface(const TrajectoryStateOnSurface& tsos1,
					  const TrajectoryStateOnSurface& tsos2) const {

  if( !tsos1.isValid() || !tsos2.isValid() ) return false;

  double phi1 = tsos1.globalPosition().phi();
  double phi2 = tsos2.globalPosition().phi();
  double eta1 = tsos1.globalPosition().eta();
  double eta2 = tsos2.globalPosition().eta();
  double dphi(fabs(Geom::Phi<double>(phi1)-Geom::Phi<double>(phi2)));
  double deta(fabs(eta1-eta2));

  float dd = 0.2;
  bool goodCoords = ( (dphi < dd) || (deta < dd) ) ? true : false;  

  return goodCoords;
}


GlobalMuonTrackMatcher::TrackCand
GlobalMuonTrackMatcher::matchMomAtIP(const TrackCand& staCand,
				     const std::vector<TrackCand>& tkTs) const{
  
  TrackCand returnVal;
  float deltaR = 1000.0;
  
  for (vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is) {
    double Eta1 = staCand.second->eta();
    double Eta2;
    if ((*is).first != 0) {
      Eta2 = (*is).first->firstMeasurement().updatedState().globalMomentum().eta();
    } else {
      Eta2 = (*is).second->eta();
    }
    double Phi1 = staCand.second->phi();
    double Phi2;
    if ((*is).first != 0) {
      Phi2 = (*is).first->firstMeasurement().updatedState().globalMomentum().phi();
    } else {
      Phi2 = (*is).second->phi();
    }
    double deltaEta = Eta1 - Eta2;
    double deltaPhi(fabs(Geom::Phi<float>(Phi1)-Geom::Phi<float>(Phi2)));
    double deltaR_tmp = sqrt(pow(deltaEta,2.) + pow(deltaPhi,2.));
    
    if (deltaR_tmp < deltaR) {
      deltaR = deltaR_tmp;
      returnVal = *is;
    }
  }    
  
  return returnVal;  
}
