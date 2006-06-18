/**
 *  Class: GlobalMuonTrajectoryBuilder
 *
 *  Description: 
 *
 *             MuonHitsOption: 0 - tracker only
 *                             1 - include all muon hits
 *                             2 - include only first muon hit(s)
 *                             3 - include only selected muon hits
 *                             4 - combined
 *
 *
 *  $Date: $
 *  $Revision: $
 *
 *  Author :
 *  N. Neumeister            Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *  porting author:
 *  C. Liu                   Purdue University
 *
**/

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonSeedCleaner.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonReFitter.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

using namespace std;
//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par) :

  theMuons(0),
  theTrajectoryBuilder(0),
  theTrajectorySmoother(0),
  theTrajectoryCleaner(0),
  theRefitter(0) {

  theDirection = static_cast<ReconstructionDirection>(par.getParameter<int>("Direction"));
  theMuonHitsOption = par.getParameter<int>("MuonHitsOption");
  thePtCut = par.getParameter<double>("ptCut");
  theProbCut = par.getParameter<double>("Chi2ProbabilityCut");
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");

  init();

}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {

  delete theRefitter;
  delete theTrajectoryCleaner;
  delete theTrajectorySmoother;
  delete theTrajectoryBuilder;
  delete theMuons;

}


//--------------
// Operations --
//--------------

//
// reconstruct muons
// 
GlobalMuonTrajectoryBuilder::MuonCollection GlobalMuonTrajectoryBuilder::muons() {

  MuonCollection result;
  return result;

}


//
// get silicon tracker tracks
//
vector<Trajectory> GlobalMuonTrajectoryBuilder::getTrackerTracks(const Muon& muon, int& nSeeds, int& nRaw, int& nSmoothed, int& nCleaned) const {
  
  // collect trajectories for all seeds
  //
  vector<Trajectory> rawResult;

  vector<Trajectory> cleanedResult(rawResult);

  return cleanedResult;

}

 
//
//  check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
//
void GlobalMuonTrajectoryBuilder::checkMuonHits(const Muon& muon, 
                                          RecHitContainer& all,
                                          RecHitContainer& first,
                                          vector<int>& hits) const {
 

}


//
//  select muon hits compatible with trajectory; check hits in chambers with showers
//
GlobalMuonTrajectoryBuilder::RecHitContainer GlobalMuonTrajectoryBuilder::selectMuonHits(const Trajectory& track, const vector<int>& hits) const {

  RecHitContainer muonRecHits;

 
  return muonRecHits;

}


//
// calculate the tail probability (-ln(P)) of a fit
//
double GlobalMuonTrajectoryBuilder::trackProbability(const Trajectory& track) const {

  int nDOF = 0;
  RecHitContainer rechits = track.recHits();
  for (RecHitContainer::const_iterator i = rechits.begin(); i != rechits.end(); ++i ) {
    if ((*i).isValid()) nDOF += (*i).dimension();
  }

  nDOF = max(nDOF - 5, 0);
  double prob = -LnChiSquaredProbability(track.chiSquared(), nDOF);

  return prob;

}


//
// find a match between one trajectory and a list of tracker trajectories 
//
GlobalMuonTrajectoryBuilder::TI GlobalMuonTrajectoryBuilder::matchTrajectories(const Trajectory& track, TC& tracks) const {

  TI match = tracks.end();

  return match;

}


//
// choose final trajectory
//
const Trajectory* GlobalMuonTrajectoryBuilder::chooseTrajectory(const vector<Trajectory*>& t) const {

  const Trajectory* result = 0;
 
  double prob0 = ( t[0] ) ? trackProbability(*t[0]) : 0.0;
  double prob1 = ( t[1] ) ? trackProbability(*t[1]) : 0.0;
  double prob2 = ( t[2] ) ? trackProbability(*t[2]) : 0.0;
  double prob3 = ( t[3] ) ? trackProbability(*t[3]) : 0.0; 

 // edm::LogInfo << "Probabilities: " << prob0 << " " << prob1 << " " << prob2 << " " << prob3 << endl;

  if ( t[1] ) result = t[1];
  if ( (t[1] == 0) && t[3] ) result = t[3];
  
  if ( t[1] && t[3] && ( (prob1 - prob3) > 0.05 )  )  result = t[3];

  if ( t[0] && t[2] && fabs(prob2 - prob0) > theProbCut ) {
 //   cout.debugOut << "select Tracker only: -log(prob) = " << prob0 << endl;
    result = t[0];
    return result;
  }

  if ( (t[1] == 0) && (t[3] == 0) && t[2] ) result = t[2];

  Trajectory* tmin = 0;
  double probmin = 0.0;
  if ( t[1] && t[3] ) {
    probmin = prob3; tmin = t[3];
    if ( prob1 < prob3 ) { probmin = prob1; tmin = t[1]; }
  }
  else if ( (t[3] == 0) && t[1] ) { 
    probmin = prob1; tmin = t[1]; 
  }
  else if ( (t[1] == 0) && t[3] ) {
    probmin = prob3; tmin = t[3]; 
  }

  if ( tmin && t[2] && ( (probmin - prob2) > 3.5 )  ) {
    result = t[2];
  }

  return result;

}


//
//
//
GlobalMuonTrajectoryBuilder::MuonCollection GlobalMuonTrajectoryBuilder::convertToRecMuons(vector<TrajForRecMuon>& refittedResult) const {

//  typedef vector<TrajForRecMuon>::const_iterator TI;
  MuonCollection result;
  //
  return result;

}


//
// print RecHits
//
void GlobalMuonTrajectoryBuilder::printHits(const RecHitContainer& hits) const {

}


//
// initialize algorithms
//
void GlobalMuonTrajectoryBuilder::init() {


}


//
// check candidates
//

void GlobalMuonTrajectoryBuilder::checkMuonCandidates(MuonCollection& candidates) const {


}


