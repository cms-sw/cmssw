/**
 *  Class: L3MuonTrajectoryBuilder
 *
 *  Description:
 *   Reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *
 *  $Date:  $
 *  $Revision:  $
 *
 *  Authors :
 *  N. Neumeister            Purdue University
 *  C. Liu                   Purdue University
 *  A. Everett               Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *
 **/

#include "RecoMuon/L3TrackFinder/interface/L3MuonTrajectoryBuilder.h"


//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

L3MuonTrajectoryBuilder::L3MuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : GlobalTrajectoryBuilderBase(par, service), theTkSeedGenerator(0), theNavigationSchool(0), theCacheId_DG(0),theCacheId_MG(0) {

  theTkTrajsAvailableFlag = par.getParameter<bool>("TkTrajectoryAvailable");  
  theFirstEvent = true;
  
  //
  // start seed generator;
  //
  string stateOnTrackerOutProp = par.getParameter<string>("StateOnTrackerBoundOutPropagator");
  ParameterSet seedGenPSet = par.getParameter<ParameterSet>("SeedGeneratorParameters");
  seedGenPSet.addParameter<string>("StateOnTrackerBoundOutPropagator",stateOnTrackerOutProp);
  std::string seedGenName = seedGenPSet.getParameter<std::string>("ComponentName");
  theTkSeedGenerator = TrackerSeedGeneratorFactory::get()->create(seedGenName, seedGenPSet);
  theTkSeedGenerator->init(GlobalTrajectoryBuilderBase::service());
  
  theTkBuilderName = par.getParameter<std::string>("TkTrackBuilder");

  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    
}


//--------------
// Destructor --
//--------------

L3MuonTrajectoryBuilder::~L3MuonTrajectoryBuilder() {
  if (theTrajectoryCleaner) delete theTrajectoryCleaner;
}

//
// get information from event
//
void L3MuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  
  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|setEvent";
    
  GlobalTrajectoryBuilderBase::setEvent(event);
    
  // DetLayer Geometry
  unsigned long long newCacheId_DG = GlobalTrajectoryBuilderBase::service()->eventSetup().get<TrackerRecoGeometryRecord>().cacheIdentifier();
  unsigned long long newCacheId_MG = GlobalTrajectoryBuilderBase::service()->eventSetup().get<IdealMagneticFieldRecord>().cacheIdentifier();
    
  if(newCacheId_DG != theCacheId_DG || newCacheId_MG != theCacheId_MG) {
    LogTrace(category) << "Tracker Reco Geometry changed!";
    theCacheId_DG = newCacheId_DG;  
    theCacheId_MG = newCacheId_MG;  
    edm::ESHandle<GeometricSearchTracker> geomSearchTracker;
    GlobalTrajectoryBuilderBase::service()->eventSetup().get<TrackerRecoGeometryRecord>().get( geomSearchTracker );  
    
    // NavigationSchool should live until its validity expires, and then DELETE
    // the NavigableLayers
    if(theNavigationSchool) delete theNavigationSchool;          
    
    theNavigationSchool = new SimpleNavigationSchool(&(*geomSearchTracker),&(*(GlobalTrajectoryBuilderBase::service()->magneticField())));  
  }
  
  // set the correct navigation
    
  if(theNavigationSchool) NavigationSetter setter( *theNavigationSchool);
    
  if (theFirstEvent) {
      
    theFirstEvent = false;
    LogInfo(category) << "Constructing a Tk Trajectory Builder";
    GlobalTrajectoryBuilderBase::service()->eventSetup().get<CkfComponentsRecord>().get(theTkBuilderName,theTkBuilder);  
  }
    
  theTkBuilder->setEvent(event);
    
  theTkSeedGenerator->setEvent(event);
  
}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer L3MuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|trajectories";

  // cut on muons with low momenta
  if ( (staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut || (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);
  addTraj(staCand);
  
  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogInfo(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";  

  // match tracker tracks to muon track
  vector<TrackCand> trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  LogInfo(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if ( trackerTracks.empty() ) return CandidateContainer();
  
  // build a combined tracker-muon MuonCandidate
  //
  // turn tkMatchedTracks into MuonCandidates
  //
  LogInfo(category) << "turn tkMatchedTracks into MuonCandidates";
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = trackerTracks.begin(); tkt != trackerTracks.end(); tkt++) {
    if ((*tkt).first != 0 && (*tkt).first->isValid()) {
      std::vector<Trajectory> refittedTk;
      Trajectory refittedTkTraj = *(*tkt).first;
      refittedTk = refitTrajectory(*(*tkt).first);
      if(refittedTk.size() == 1) refittedTkTraj = refittedTk.front();
      
      MuonCandidate* muonCand = new MuonCandidate( 0 ,staCand.second,(*tkt).second, new Trajectory(refittedTkTraj));
      tkTrajs.push_back(muonCand);
      LogTrace(category) << "tpush";

    }
  }
  
  if ( tkTrajs.empty() )  {
    LogInfo(category) << "tkTrajs empty";
    return CandidateContainer();
  }
  
  CandidateContainer result = build(staCand, tkTrajs);  
  LogInfo(category) << "Found "<< result.size() << " L3Muons from one L2Cand";

  // free memory
  if ( staCandIn.first == 0) delete staCand.first;

  for( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); ++it) {
    if ( (*it)->trajectory() ) delete (*it)->trajectory();
    if ( (*it)->trackerTrajectory() ) delete (*it)->trackerTrajectory();
    if ( *it ) delete (*it);
  }
  tkTrajs.clear();  

  if ( !theTkTrajsAvailableFlag ) {
    for ( vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
      delete (*is).first;   
    }
  }

  return result;
  
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<L3MuonTrajectoryBuilder::TrackCand> L3MuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) const {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;  

  // Tracks not available, make seeds and trajectories

  LogTrace(category) << "Making Seeds";
  
  std::vector<TrajectorySeed> tkSeeds; 
  TC allTkTrajs;
  
  RectangularEtaPhiTrackingRegion region = defineRegionOfInterest((staCand.second));
  theTkSeedGenerator->trackerSeeds(staCand, region, tkSeeds);
  
  LogTrace(category) << "Found " << tkSeeds.size() << " tracker seeds";
    
  allTkTrajs = makeTrajsFromSeeds(tkSeeds);
  
  for (TC::const_iterator tt=allTkTrajs.begin();tt!=allTkTrajs.end();++tt){
    tkCandColl.push_back(TrackCand(new Trajectory(*tt),reco::TrackRef()));
  } 
  
  LogTrace(category) << "Found " << tkCandColl.size() << " tkCands from seeds";

  return tkCandColl;

}


//
// build a tracker Trajectory from a seed
//
L3MuonTrajectoryBuilder::TC L3MuonTrajectoryBuilder::makeTrajsFromSeeds(const vector<TrajectorySeed>& tkSeeds) const {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|makeTrajsFromSeeds";
  TC result;
  
  LogInfo(category) << "Tracker Seeds from L2/STA Muon: " << tkSeeds.size();
  
  int nseed = 0;
  vector<Trajectory> rawResult;
  std::vector<TrajectorySeed>::const_iterator seed;
  for (seed = tkSeeds.begin(); seed != tkSeeds.end(); ++seed) {
    nseed++;
    LogTrace(category) << "Building a trajectory from seed " << nseed;
    
    TC tkTrajs;

    tkTrajs = theTkBuilder->trajectories(*seed);

    LogTrace(category) << "Trajectories from Seed " << tkTrajs.size();
    
    theTrajectoryCleaner->clean(tkTrajs);
    
    for(vector<Trajectory>::const_iterator it=tkTrajs.begin();
	it!=tkTrajs.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
      }
    }
    LogTrace(category) << "Trajectories from Seed after cleaning " << rawResult.size();
    
  }

  theTrajectoryCleaner->clean(rawResult);
  
  for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
       itraw != rawResult.end(); itraw++) {
    if((*itraw).isValid()) result.push_back( *itraw);
  }
 

  LogInfo(category) << "Trajectories from all seeds " << result.size();
  return result;

}


