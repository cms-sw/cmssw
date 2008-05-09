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
 *  $Date: 2008/02/26 05:15:35 $
 *  $Revision: 1.8 $
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
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

L3MuonTrajectoryBuilder::L3MuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service) : GlobalTrajectoryBuilderBase(par, service), theTkSeedGenerator(0) {

  theFirstEvent = true;
  
  //
  // start seed generator;
  //
   ParameterSet seedGenPSet = par.getParameter<ParameterSet>("SeedGeneratorParameters");
  std::string seedGenName = seedGenPSet.getParameter<std::string>("ComponentName");
  theTkSeedGenerator = TrackerSeedGeneratorFactory::get()->create(seedGenName, seedGenPSet);
  theTkSeedGenerator->init(GlobalTrajectoryBuilderBase::service());
  
  theTkBuilderName = par.getParameter<std::string>("TkTrackBuilder");

  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    

  theSeedName = par.getParameter<edm::InputTag>("l3SeedLabel");
  theTkCollName = par.getParameter<edm::InputTag>("tkTrajLabel");

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
    
    
  if (theFirstEvent) {
      
    theFirstEvent = false;
    LogInfo(category) << "Constructing a Tk Trajectory Builder";
    GlobalTrajectoryBuilderBase::service()->eventSetup().get<CkfComponentsRecord>().get(theTkBuilderName,theTkBuilder);  
  }
    
  theTkBuilder->setEvent(event);
    
  theTkSeedGenerator->setEvent(event);

  theSeedsAvailable = event.getByLabel(theSeedName,theSeedCollection);
  theTrajsAvailable = event.getByLabel(theTkCollName,theTkTrajCollection);
  theTkCandsAvailable = event.getByLabel(theTkCollName,theTkTrackCandCollection);
  
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

  vector<TrackCand> trackerTracks;

  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogInfo(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";  
  
  // match tracker tracks to muon track
  trackerTracks = trackMatcher()->match(staCand, regionalTkTracks);
  
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
      
      LogDebug(category)<< "seedRef " << refittedTkTraj.seedRef().isNonnull();

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

  for ( vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
    delete (*is).first;   
  }
  
  return result;
  
}

//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<L3MuonTrajectoryBuilder::TrackCand> L3MuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) {

  const std::string category = "Muon|RecoMuon|L3MuonTrajectoryBuilder|makeTkCandCollection";

  vector<TrackCand> tkCandColl;  

  if (theTrajsAvailable) {
    LogDebug(category) << "Found " << theTkTrajCollection->size() <<" tkCands";
    for (TC::const_iterator tt=theTkTrajCollection->begin();tt!=theTkTrajCollection->end();++tt){
      tkCandColl.push_back(TrackCand(new Trajectory(*tt),reco::TrackRef()));
      LogDebug(category)<< "seedRef " << tkCandColl.back().first->seedRef().isNonnull();
    } 
    LogTrace(category) << "Found " << tkCandColl.size() << " tkCands from seeds";
    return tkCandColl;
  }
  
  // Tracks not available, make seeds and trajectories
  // std::vector<L3MuonTrajectorySeed> useSeeds;
  std::vector<TrajectorySeed> tkSeeds;
  if( theSeedsAvailable ) {
    L3MuonTrajectorySeedCollection::const_iterator l3Seed;
    for(l3Seed=theSeedCollection->begin(); l3Seed != theSeedCollection->end(); ++l3Seed) {
      const reco::TrackRef & l2FromSeed = l3Seed->l2Track();
      if(staCand.second == l2FromSeed) tkSeeds.push_back(*l3Seed);
    }    
  } else {
    
    LogTrace(category) << "Making Seeds";
    
    RectangularEtaPhiTrackingRegion region = defineRegionOfInterest((staCand.second));
    theTkSeedGenerator->trackerSeeds(staCand, region, tkSeeds);    
  }

  LogTrace(category) << "Found " << tkSeeds.size() << " tracker seeds";

  TC allTkTrajs = makeTrajsFromSeeds(tkSeeds);
  
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

  // retrieve navigation school
  edm::ESHandle<NavigationSchool> nav;
  GlobalTrajectoryBuilderBase::service()->eventSetup().get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  // set the correct navigation
  NavigationSetter setter(*nav.product());

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


