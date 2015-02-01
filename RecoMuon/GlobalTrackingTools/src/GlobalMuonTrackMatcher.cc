/**
 *  Class: GlobalMuonTrackMatcher
 *
 * 
 *  
 *  \author Chang Liu - Purdue University
 *  \author Norbert Neumeister - Purdue University
 *  \author Adam Everett - Purdue University
 *  \author Edwin Antillon - Purdue University
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
  theMinP = par.getParameter<double>("MinP");
  theMinPt = par.getParameter<double>("MinPt");
  thePt_threshold1 = par.getParameter<double>("Pt_threshold1");
  thePt_threshold2 = par.getParameter<double>("Pt_threshold2");
  theEta_threshold= par.getParameter<double>("Eta_threshold");
  theChi2_1= par.getParameter<double>("Chi2Cut_1");
  theChi2_2= par.getParameter<double>("Chi2Cut_2");
  theChi2_3= par.getParameter<double>("Chi2Cut_3");
  theLocChi2= par.getParameter<double>("LocChi2Cut");
  theDeltaD_1= par.getParameter<double>("DeltaDCut_1");
  theDeltaD_2= par.getParameter<double>("DeltaDCut_2");
  theDeltaD_3= par.getParameter<double>("DeltaDCut_3");
  theDeltaR_1= par.getParameter<double>("DeltaRCut_1");
  theDeltaR_2= par.getParameter<double>("DeltaRCut_2");
  theDeltaR_3= par.getParameter<double>("DeltaRCut_3");
  theQual_1= par.getParameter<double>("Quality_1");
  theQual_2= par.getParameter<double>("Quality_2");
  theQual_3= par.getParameter<double>("Quality_3");
  theOutPropagatorName = par.getParameter<string>("Propagator");

}


//
// destructor
//
GlobalMuonTrackMatcher::~GlobalMuonTrackMatcher() {

}


//
// check if two tracks are compatible with the *tight* criteria
//
bool 
GlobalMuonTrackMatcher::matchTight(const TrackCand& sta,
                              const TrackCand& track) const {

  std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair
      = convertToTSOSMuHit(sta,track);

  double chi2 = match_Chi2(tsosPair.first,tsosPair.second);
  if ( chi2 > 0. && chi2 < theChi2_2 ) return true;

  double distance = match_d(tsosPair.first,tsosPair.second);
  if ( distance > 0. && distance < theDeltaD_2 ) return true;

  //double deltaR = match_Rpos(tsosPair.first,tsosPair.second);
  //if ( deltaR > 0. && deltaR < theDeltaR_3 ) return true;

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
  else if ( matchOption == 3 ) {
    return match_dist(tsosPair.first,tsosPair.second);
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
    if ( chi2 > 0. && chi2 <= minChi2 ) {
      minChi2 = chi2;
      result = is;
    }
    
  }
  
  return result;
  
}


//
// choose a vector of tracks from a TrackCollection that are compatible
// with a given standalone track. The order of checks for compatability 
// * for low momentum: use chi2 selection 
// * high momentum: use direction or local position 
//
vector<GlobalMuonTrackMatcher::TrackCand>
GlobalMuonTrackMatcher::match(const TrackCand& sta, 
                              const vector<TrackCand>& tracks) const {
  const string category = "GlobalMuonTrackMatcher";
  
  vector<TrackCand> result;
  
  if ( tracks.empty() ) return result;
  
  typedef std::pair<TrackCand, TrajectoryStateOnSurface> TrackCandWithTSOS;
  vector<TrackCandWithTSOS> cands;
  int iiTk = 1;
  TrajectoryStateOnSurface muonTSOS;

  LogTrace(category) << "   ***" << endl << "STA Muon pT "<< sta.second->pt(); 
  LogTrace(category) << "   Tk in Region " << tracks.size() << endl;

  for (vector<TrackCand>::const_iterator is = tracks.begin(); is != tracks.end(); ++is,iiTk++) {
    // propagate to a common surface 
    std::pair<TrajectoryStateOnSurface, TrajectoryStateOnSurface> tsosPair = convertToTSOSMuHit(sta,*is);
    LogTrace(category) << "    Tk " << iiTk << " of " << tracks.size() << "  ConvertToMuHitSurface muon isValid " << tsosPair.first.isValid() << " tk isValid " << tsosPair.second.isValid() << endl;
    if(tsosPair.first.isValid()) muonTSOS = tsosPair.first;
    cands.push_back(TrackCandWithTSOS(*is,tsosPair.second));
  }
  
  // initialize variables
  double min_chisq = 999999;
  double min_d = 999999;
  double min_de= 999999;
  double min_r_pos = 999999;
  std::vector<bool> passes(cands.size(),false);
  int jj=0;

  int iTkCand = 1;
  for (vector<TrackCandWithTSOS>::const_iterator ii = cands.begin(); ii != cands.end(); ++ii,jj++,iTkCand++) {
    
    // tracks that are able not able propagate to a common surface
    if(!muonTSOS.isValid() || !(*ii).second.isValid()) continue;
    
    // calculate matching variables
    double distance = match_d(muonTSOS,(*ii).second);
    double chi2 = match_Chi2(muonTSOS,(*ii).second);
    double loc_chi2 = match_dist(muonTSOS,(*ii).second);
    double deltaR = match_Rpos(muonTSOS,(*ii).second);

    LogTrace(category) << "   iTk " << iTkCand << " of " << cands.size() << " eta " << (*ii).second.globalPosition().eta() << " phi " << (*ii).second.globalPosition().phi() << endl; 
    LogTrace(category) << "    distance " << distance << " distance cut " << " " << endl;
    LogTrace(category) << "    chi2 " << chi2 << " chi2 cut " << " " << endl;
    LogTrace(category) << "    loc_chi2 " << loc_chi2 << " locChi2 cut " << " " << endl;
    LogTrace(category) << "    deltaR " << deltaR << " deltaR cut " << " " << endl;   
    
    if( (*ii).second.globalMomentum().perp()<thePt_threshold1){
      LogTrace(category) << "    Enters  a1" << endl;

      if( ( chi2>0 && fabs((*ii).second.globalMomentum().eta())<theEta_threshold && chi2<theChi2_1 ) || (distance>0 && distance/(*ii).first.second->pt()<theDeltaD_1 && loc_chi2>0 && loc_chi2<theLocChi2) ){
	LogTrace(category) << "    Passes a1" << endl;
        result.push_back((*ii).first);
        passes[jj]=true;
      }
    }
    if( (passes[jj]==false) && (*ii).second.globalMomentum().perp()<thePt_threshold2){
      LogTrace(category) << "    Enters a2" << endl;
      if( ( chi2>0 && chi2< theChi2_2 ) || (distance>0 && distance<theDeltaD_2) ){
	LogTrace(category) << "    Passes a2" << endl;
	result.push_back((*ii).first);
	passes[jj] = true;
      }
    }else{
      LogTrace(category) << "    Enters a3" << endl;
      if( distance>0 && distance<theDeltaD_3 && deltaR>0 && deltaR<theDeltaR_1){
	LogTrace(category) << "    Passes a3" << endl;
	result.push_back((*ii).first);
        passes[jj]=true;
      }
    }
    
    if(passes[jj]){
      if(distance<min_d) min_d = distance;
      if(loc_chi2<min_de) min_de = loc_chi2;
      if(deltaR<min_r_pos) min_r_pos = deltaR;
      if(chi2<min_chisq) min_chisq = chi2;
    }

  }
  
  // re-initialize mask counter
  jj=0;
  
  if ( result.empty() ) {
    LogTrace(category) << "   Stage 1 returned 0 results";
    for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is,jj++) {
      double deltaR = match_Rpos(muonTSOS,(*is).second);

      if (muonTSOS.isValid() && (*is).second.isValid()) {
	// check matching between tracker and muon tracks using dEta cut looser then dPhi cut 
	LogTrace(category) << "    Stage 2 deltaR " << deltaR << " deltaEta " << fabs((*is).second.globalPosition().eta()-muonTSOS.globalPosition().eta()<1.5*theDeltaR_2) << " deltaPhi " << (fabs(deltaPhi((*is).second.globalPosition().phi(),muonTSOS.globalPosition().phi()))<theDeltaR_2) << endl;
        
	if(fabs((*is).second.globalPosition().eta()-muonTSOS.globalPosition().eta())<1.5*theDeltaR_2
	   &&fabs(deltaPhi((*is).second.globalPosition().phi(),muonTSOS.globalPosition().phi()))<theDeltaR_2){
	  result.push_back((*is).first);
	  passes[jj]=true;
	}
      }
      
      if(passes[jj]){
        double distance = match_d(muonTSOS,(*is).second);
        double chi2 = match_Chi2(muonTSOS,(*is).second);
        double loc_chi2 = match_dist(muonTSOS,(*is).second);
        if(distance<min_d) min_d = distance;
        if(loc_chi2<min_de) min_de = loc_chi2;
        if(deltaR<min_r_pos) min_r_pos = deltaR;
        if(chi2<min_chisq) min_chisq = chi2;
	
      }
      
    }
    
  }  

  for(vector<TrackCand>::const_iterator iTk=result.begin();
      iTk != result.end(); ++iTk) {
    LogTrace(category) << "   -----" << endl 
			      << "selected pt " << iTk->second->pt() 
			      << " eta " << iTk->second->eta() 
			      << " phi " << iTk->second->phi() << endl; 
  }

  if(result.size()<2)
    return result;
  else
    result.clear();
  
  LogTrace(category) << "   Cleaning matched candiates" << endl;

  // re-initialize mask counter
  jj=0;
  
  
  for (vector<TrackCandWithTSOS>::const_iterator is = cands.begin(); is != cands.end(); ++is,jj++) {
    
    if(!passes[jj]) continue;
    
    double distance = match_d(muonTSOS,(*is).second);
    double chi2 = match_Chi2(muonTSOS,(*is).second);
    //unused    double loc_chi2 = match_dist(muonTSOS,(*is).second);
    double deltaR = match_Rpos(muonTSOS,(*is).second);
    
    // compute quality as the relative ratio to the minimum found for each variable
    
    int qual = (int)(chi2/min_chisq + distance/min_d + deltaR/min_r_pos);
    int n_min = ((chi2/min_chisq==1)?1:0) + ((distance/min_d==1)?1:0) + ((deltaR/min_r_pos==1)?1:0);
    
    if(n_min == 3){
      result.push_back((*is).first);
    }
    
    if(n_min == 2 && qual < theQual_1 ){
      result.push_back((*is).first);
    }
    
    if(n_min == 1 && qual < theQual_2 ){
      result.push_back((*is).first);
    }
    
    if(n_min == 0 && qual < theQual_3 ){
      result.push_back((*is).first);
    }
    
  }

  for(vector<TrackCand>::const_iterator iTk=result.begin();
      iTk != result.end(); ++iTk) {
    LogTrace(category) << "   -----" << endl 
			      << "selected pt " << iTk->second->pt() 
			      << " eta " << iTk->second->eta() 
			      << " phi " << iTk->second->phi() << endl; 
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
      
      outerTkTsos = trajectoryStateTransform::outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
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
    LogTrace(category) << "Propagating to same tracker surface (Mu):" << same1;
    if ( !same1 ) {
      if ( tkTsosFromTk.isValid() ) newTkTsosFromMu = theService->propagator(theOutPropagatorName)->propagate(impactMuTSOS,tkTsosFromTk.surface());
      same2 = samePlane(newTkTsosFromMu,tkTsosFromTk);
      LogTrace(category) << "Propagating to same tracker surface (Tk):" << same2;
    }
    if (same1) tkTsosFromTk = newTkTsosFromTk;
    else if (same2) tkTsosFromMu = newTkTsosFromMu;
    else  {
      LogTrace(category) << "Could not propagate Muon and Tracker track to the same tracker bound!";
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
  TrajectoryStateOnSurface outerTkTsos,innerTkTsos;
  if ( tkCand.second.isNonnull() ) {
    // make sure the tracker track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt ) ) {
      TrajectoryStateOnSurface innerTkTsos;
      
      outerTkTsos = trajectoryStateTransform::outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
      innerTkTsos = trajectoryStateTransform::innerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
      // for cosmics, outer-most referst to last traversed layer
      if ( (innerMuTSOS.globalPosition() -  outerTkTsos.globalPosition()).mag() > (innerMuTSOS.globalPosition() -  innerTkTsos.globalPosition()).mag() )
	outerTkTsos = innerTkTsos;
      
    }
  }
    
  if ( !innerMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogTrace(category) << "A TSOS validity problem! MuTSOS " << innerMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }
  
  const Surface& refSurface = innerMuTSOS.surface();
  TrajectoryStateOnSurface tkAtMu = theService->propagator(theOutPropagatorName)->propagate(*outerTkTsos.freeState(),refSurface);
  
  if ( !tkAtMu.isValid() ) {
    LogTrace(category) << "Could not propagate Muon and Tracker track to the same muon hit surface!";
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
  TrajectoryStateOnSurface innerMuTSOS = muTT.innermostMeasurementState();
  
  TrajectoryStateOnSurface outerTkTsos,innerTkTsos;
  if ( tkCand.second.isNonnull() ) {
    // make sure the tracker track has enough momentum to reach the muon chambers
    if ( !(tkCand.second->p() < theMinP || tkCand.second->pt() < theMinPt )) {
      
      outerTkTsos = trajectoryStateTransform::outerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
      innerTkTsos = trajectoryStateTransform::innerStateOnSurface(*tkCand.second,*theService->trackingGeometry(),&*theService->magneticField());
      
      // for cosmics, outer-most referst to last traversed layer
      if ( (innerMuTSOS.globalPosition() -  outerTkTsos.globalPosition()).mag() > (innerMuTSOS.globalPosition() -  innerTkTsos.globalPosition()).mag() )
	outerTkTsos = innerTkTsos;
      
    }
  }

  if ( !impactMuTSOS.isValid() || !outerTkTsos.isValid() ) {
    LogTrace(category) << "A TSOS validity problem! MuTSOS " << impactMuTSOS.isValid() << " TkTSOS " << outerTkTsos.isValid();
    return pair<TrajectoryStateOnSurface,TrajectoryStateOnSurface>(empty,empty);
  }

  const Surface& refSurface = outerTkTsos.surface();
  TrajectoryStateOnSurface muAtTk = theService->propagator(theOutPropagatorName)->propagate(*impactMuTSOS.freeState(),refSurface);
  
  if ( !muAtTk.isValid() ) {
    LogTrace(category) << "Could not propagate Muon and Tracker track to the same tracker hit surface!";
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

  if ( fabs(match_D(tsos1,tsos2) - match_d(tsos1,tsos2)) > 0.1 ) return false;

  const float maxtilt = 0.999;
  const float maxdist = 0.01; // in cm

  auto p1(tsos1.surface().tangentPlane(tsos1.localPosition()));
  auto p2(tsos2.surface().tangentPlane(tsos2.localPosition()));

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
  //LogTrace(category) << "match_Chi2 sanity check: " << tsos1.isValid() << " " << tsos2.isValid();
  if ( !tsos1.isValid() || !tsos2.isValid() ) return -1.;
  
  AlgebraicVector5 v(tsos1.localParameters().vector() - tsos2.localParameters().vector());
  AlgebraicSymMatrix55 m(tsos1.localError().matrix() + tsos2.localError().matrix());
  
  //LogTrace(category) << "match_Chi2 vector v " << v;

  bool ierr = !m.Invert();
 
  if ( ierr ) { 
    edm::LogInfo(category) << "Error inverting covariance matrix";
    return -1;
  }
 
  double est = ROOT::Math::Similarity(v,m);
 
  //LogTrace(category) << "Chi2 " << est;

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


//
// calculate the chi2 of the distance in local position of two 
// trajectory states including local errors
//
double
GlobalMuonTrackMatcher::match_dist(const TrajectoryStateOnSurface& sta, 
				   const TrajectoryStateOnSurface& tk) const {
 
   const string category = "GlobalMuonTrackMatcher";
   
   if ( !sta.isValid() || !tk.isValid() ) return -1;
 
   AlgebraicMatrix22 m;
   m(0,0) = tk.localError().positionError().xx() + sta.localError().positionError().xx();
   m(1,0) = m(0,1) = tk.localError().positionError().xy() + sta.localError().positionError().xy();
   m(1,1) = tk.localError().positionError().yy() + sta.localError().positionError().yy();
 
   AlgebraicVector2 v;
   v[0] = tk.localPosition().x() - sta.localPosition().x();
   v[1] = tk.localPosition().y() - sta.localPosition().y();
 
   if ( !m.Invert() ) {
     LogTrace(category) << "Error inverting local matrix ";
     return -1;
   }
 
   return ROOT::Math::Similarity(v,m);

}
