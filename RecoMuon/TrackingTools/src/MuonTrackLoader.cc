
/** \class MuonTrackLoader
 *  Class to load the product in the event
 *


 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"


using namespace edm;
using namespace std;
using namespace reco;

std::vector<const TrackingRecHit*> MuonTrackLoader::unpackHit(const TrackingRecHit &hit)
{
    // get rec hit det id and rec hit type
    DetId id = hit.geographicalId();
    uint16_t detid = id.det();

    std::vector<const TrackingRecHit*> hits;

    if (detid == DetId::Tracker) {
    	hits.push_back(&hit);
    } else if (detid == DetId::Muon) {
        uint16_t subdet = id.subdetId();
        if (subdet == (uint16_t) MuonSubdetId::DT) {
            if (hit.dimension() == 1) { // DT rechit (granularity 2)
                hits.push_back(&hit);
            } else if (hit.dimension() > 1) { // 4D segment (granularity 0).
                // Both 2 and 4 dim cases. MB4s have 2D, but formatted in 4D segment
                // 4D --> 2D
                std::vector<const TrackingRecHit*> seg2D = hit.recHits();
                // load 1D hits (2D --> 1D)
                for (std::vector<const TrackingRecHit*>::const_iterator it = seg2D.begin(); it != seg2D.end(); ++it) {
                    std::vector<const TrackingRecHit*> hits1D = (*it)->recHits();
                    copy(hits1D.begin(), hits1D.end(), back_inserter(hits));
                }
            }
        } else if (subdet == (uint16_t) MuonSubdetId::CSC) {
            if (hit.dimension() == 2) { // CSC rechit (granularity 2)
                hits.push_back(&hit);
            } else if (hit.dimension() == 4) { // 4D segment (granularity 0)
                // load 2D hits (4D --> 1D)
                hits = hit.recHits();
            }
        } else if (subdet == (uint16_t) MuonSubdetId::RPC) {
            hits.push_back(&hit);
        }
    }
    return hits;
}

// constructor (obsolete, should use eventSetUp not service..)
MuonTrackLoader::MuonTrackLoader(ParameterSet &parameterSet, edm::ConsumesCollector& iC, const MuonServiceProxy *service): 
  theService(service){


  // option to do or not the smoothing step.
  // the trajectories which are passed to the track loader are supposed to be non-smoothed
  theSmoothingStep = parameterSet.getParameter<bool>("DoSmoothing");
  if(theSmoothingStep)
    theSmootherName = parameterSet.getParameter<string>("Smoother");  

  theTrackerRecHitBuilderName = parameterSet.getParameter<std::string>("TTRHBuilder");
  
  // update at vertex
  theUpdatingAtVtx = parameterSet.getParameter<bool>("VertexConstraint");

  // beam spot input tag
  theBeamSpotInputTag = parameterSet.getParameter<edm::InputTag>("beamSpot");
  theBeamSpotToken = iC.consumes<reco::BeamSpot>(theBeamSpotInputTag);


  // Flag to put the trajectory into the event
  theTrajectoryFlag = parameterSet.getUntrackedParameter<bool>("PutTrajectoryIntoEvent",true);

  theL2SeededTkLabel = parameterSet.getUntrackedParameter<string>("MuonSeededTracksInstance",string());
  
  ParameterSet updatorPar = parameterSet.getParameter<ParameterSet>("MuonUpdatorAtVertexParameters");
  theUpdatorAtVtx = new MuonUpdatorAtVertex(updatorPar,service);

  thePutTkTrackFlag = parameterSet.getUntrackedParameter<bool>("PutTkTrackIntoEvent",false);
  theSmoothTkTrackFlag = parameterSet.getUntrackedParameter<bool>("SmoothTkTrack",false);
  theAllowNoVtxFlag = parameterSet.getUntrackedParameter<bool>("AllowNoVertex",false);
}

MuonTrackLoader::~MuonTrackLoader(){
  if(theUpdatorAtVtx) delete theUpdatorAtVtx;
}

OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer& trajectories,
			    Event& event, const TrackerTopology& ttopo, const string& instance, bool reallyDoSmoothing) {
  std::vector<bool> dummyVecBool;
  return loadTracks(trajectories, event, dummyVecBool, ttopo, instance, reallyDoSmoothing);
}
  
OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer& trajectories,
			    Event& event,  std::vector<bool>& tkBoolVec, 
                            const TrackerTopology& ttopo,
			    const string& instance, bool reallyDoSmoothing) {
  
  const bool doSmoothing = theSmoothingStep && reallyDoSmoothing;
  
  const string metname = "Muon|RecoMuon|MuonTrackLoader";
  
  // the track collectios; they will be loaded in the event  
  auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection());
  // ... and its reference into the event
  reco::TrackRefProd trackCollectionRefProd = event.getRefBeforePut<reco::TrackCollection>(instance);
  
  // track collection for the tracks updated at vertex
  auto_ptr<reco::TrackCollection> updatedAtVtxTrackCollection(new reco::TrackCollection());
  // ... and its (eventually) reference into the event
  reco::TrackRefProd trackUpdatedCollectionRefProd;
  if(theUpdatingAtVtx)  trackUpdatedCollectionRefProd = event.getRefBeforePut<reco::TrackCollection>(instance+"UpdatedAtVtx");
  
  // Association map between updated and non updated at vtx tracks
  auto_ptr<reco:: TrackToTrackMap> trackToTrackmap(new reco::TrackToTrackMap(trackCollectionRefProd, trackUpdatedCollectionRefProd));
  
  // the track extra collection, it will be loaded in the event  
  auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );
  // ... and its reference into the event
  reco::TrackExtraRefProd trackExtraCollectionRefProd = event.getRefBeforePut<reco::TrackExtraCollection>(instance);
  
  // the rechit collection, it will be loaded in the event  
  auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );
  // ... and its reference into the event
  TrackingRecHitRefProd recHitCollectionRefProd = event.getRefBeforePut<TrackingRecHitCollection>(instance);
  
  // Collection of Trajectory
  auto_ptr<vector<Trajectory> > trajectoryCollection(new vector<Trajectory>);
  
  // don't waste any time...
  if ( trajectories.empty() ) { 
    event.put(recHitCollection,instance);
    event.put(trackExtraCollection,instance);
    if(theTrajectoryFlag) {
      event.put(trajectoryCollection,instance);

      // Association map between track and trajectory
      std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
      event.put( trajTrackMap, instance );
    }
    if(theUpdatingAtVtx){
      event.put(trackToTrackmap);
      event.put(updatedAtVtxTrackCollection,instance+"UpdatedAtVtx");
    }
    return event.put(trackCollection,instance);
  }
  
  edm::Handle<reco::BeamSpot> beamSpot;
  event.getByToken(theBeamSpotToken, beamSpot);

  LogTrace(metname) << "Create the collection of Tracks";
  
  reco::TrackRef::key_type trackIndex = 0;
  reco::TrackRef::key_type trackUpdatedIndex = 0;
  
  reco::TrackExtraRef::key_type trackExtraIndex = 0;
  TrackingRecHitRef::key_type recHitsIndex = 0;
  
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;
  
  if(doSmoothing) {
    edm::ESHandle<TrajectorySmoother> aSmoother;
    theService->eventSetup().get<TrajectoryFitter::Record>().get(theSmootherName,aSmoother);
    theSmoother.reset(aSmoother->clone());
    edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
    theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    theTrackerRecHitBuilder.product();
    hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(theTrackerRecHitBuilder.product())->cloner();
    theSmoother->setHitCloner(&hitCloner);
  }  


  unsigned int tjCnt = 0;
  for(TrajectoryContainer::const_iterator rawTrajectory = trajectories.begin();
      rawTrajectory != trajectories.end(); ++rawTrajectory, ++tjCnt){
    
    Trajectory &trajectory = **rawTrajectory;
    
    if(doSmoothing){
      vector<Trajectory> trajectoriesSM = theSmoother->trajectories(**rawTrajectory);
      
      if(!trajectoriesSM.empty()) {
	const edm::RefToBase<TrajectorySeed> tmpSeedRef = (**rawTrajectory).seedRef();
	trajectory = trajectoriesSM.front();
        trajectory.setSeedRef(tmpSeedRef);
	LogDebug(metname) << "theSeedRef.isNonnull " << trajectory.seedRef().isNonnull();
      } else
	LogInfo(metname)<<"The trajectory has not been smoothed!"<<endl;
    }
    
    if(theTrajectoryFlag) {
      trajectoryCollection->push_back(trajectory);
      iTjRef++;
    }    
    
    // get the transient rechit from the trajectory
    Trajectory::RecHitContainer transHits = trajectory.recHits();
    
    if ( trajectory.direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());
    
    // build the "bare" track from the trajectory.
    // This track has the parameters defined at PCA (no update)
    pair<bool,reco::Track> resultOfTrackExtrapAtPCA = buildTrackAtPCA(trajectory, *beamSpot);
    
    // Check if the extrapolation went well    
    if(!resultOfTrackExtrapAtPCA.first) {
      delete *rawTrajectory;
      continue;
    }
    
    // take the "bare" track at PCA
    reco::Track &track = resultOfTrackExtrapAtPCA.second;
    
    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( trajectory );
    
    // get the TrackExtraRef (persitent reference of the track extra)
    reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );
    
    // set the persistent track-extra reference to the Track
    track.setExtra(trackExtraRef);
    
    // build the updated-at-vertex track, starting from the previous track
    pair<bool,reco::Track> updateResult(false,reco::Track());
    
    if(theUpdatingAtVtx){
      // build the "bare" track UPDATED at vtx
      updateResult = buildTrackUpdatedAtPCA(track, *beamSpot);

      if(!updateResult.first) ++trackIndex;
      else{
	
	// set the persistent track-extra reference to the Track
	updateResult.second.setExtra(trackExtraRef);
	
	// Fill the map
	trackToTrackmap->insert(reco::TrackRef(trackCollectionRefProd,trackIndex++),
				reco::TrackRef(trackUpdatedCollectionRefProd,trackUpdatedIndex++));
      }
    }
    
    // Fill the track extra with the rec hit (persistent-)reference
    unsigned int nHitsAdded = 0;
    for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin(); recHit != transHits.end(); ++recHit) {
      TrackingRecHit *singleHit = (**recHit).hit()->clone();
      std::vector<const TrackingRecHit*> hits = MuonTrackLoader::unpackHit(*singleHit);
      for (std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it) {
          if unlikely(!track.appendHitPattern(**it, ttopo)){
              break;
          }
      }
      
      if(theUpdatingAtVtx && updateResult.first){
          std::vector<const TrackingRecHit*> hits = MuonTrackLoader::unpackHit(*singleHit);
          for (std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it) {
              if unlikely(!updateResult.second.appendHitPattern(**it, ttopo)){
                  break;
              }
          }
      }
      recHitCollection->push_back( singleHit );  
      ++nHitsAdded;
    }
    // set the TrackingRecHitRef (persitent reference of the tracking rec hits)
    trackExtra.setHits(recHitCollectionRefProd, recHitsIndex, nHitsAdded);
    recHitsIndex +=nHitsAdded;

    // fill the TrackExtraCollection
    trackExtraCollection->push_back(trackExtra);
    
    // fill the TrackCollection
    trackCollection->push_back(track);
    iTkRef++;
    LogTrace(metname) << "Debug Track being loaded pt "<< track.pt();
    // fill the TrackCollection updated at vtx
    if(theUpdatingAtVtx && updateResult.first) 
      updatedAtVtxTrackCollection->push_back(updateResult.second);
    
    // We don't need the original trajectory anymore.
    // It has been copied by value in the trajectoryCollection, if 
    // it is required to put it into the event.
    delete *rawTrajectory;

    if(tkBoolVec.size()>tjCnt) tkBoolVec[tjCnt] = true;
    if(theTrajectoryFlag) tjTkMap[iTjRef-1] = iTkRef-1;
  }
  

  
  // Put the Collections in the event
  LogTrace(metname) << "put the Collections in the event";
  event.put(recHitCollection,instance);
  event.put(trackExtraCollection,instance);

  OrphanHandle<reco::TrackCollection> returnTrackHandle;
  OrphanHandle<reco::TrackCollection> nonUpdatedHandle;
  if(theUpdatingAtVtx){
    nonUpdatedHandle = event.put(trackCollection,instance);
    event.put(trackToTrackmap);
    returnTrackHandle = event.put(updatedAtVtxTrackCollection,instance+"UpdatedAtVtx");
  }
  else {
    returnTrackHandle = event.put(trackCollection,instance);
    nonUpdatedHandle = returnTrackHandle;
  }

  if ( theTrajectoryFlag ) {
    OrphanHandle<std::vector<Trajectory> > rTrajs = event.put(trajectoryCollection,instance);

    // Association map between track and trajectory
    std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection(rTrajs, nonUpdatedHandle) );
  
    // Now Create traj<->tracks association map
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
          i != tjTkMap.end(); i++ ) {
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
                            edm::Ref<reco::TrackCollection>( nonUpdatedHandle, (*i).second ) );
    }
    event.put( trajTrackMap, instance );
  }
  
  return returnTrackHandle;
}

OrphanHandle<reco::MuonTrackLinksCollection> 
MuonTrackLoader::loadTracks(const CandidateContainer& muonCands,
			    Event& event,
                            const TrackerTopology& ttopo) {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";
  
  // the muon collection, it will be loaded in the event
  auto_ptr<reco::MuonTrackLinksCollection> trackLinksCollection(new reco::MuonTrackLinksCollection());
  
  // don't waste any time...
  if ( muonCands.empty() ) {
    auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );
    auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );
    auto_ptr<reco::TrackCollection> trackCollection( new reco::TrackCollection() );

    event.put(recHitCollection);
    event.put(trackExtraCollection);
    event.put(trackCollection);

    //need to also put the tracker tracks collection if requested
    if(thePutTkTrackFlag){
      //will take care of putting nothing in the event but the empty collection
      TrajectoryContainer trackerTrajs;
      loadTracks(trackerTrajs, event, ttopo, theL2SeededTkLabel, theSmoothTkTrackFlag);
    } 

    return event.put(trackLinksCollection);
  }
  
  // get combined Trajectories
  TrajectoryContainer combinedTrajs;
  TrajectoryContainer trackerTrajs;
  for (CandidateContainer::const_iterator it = muonCands.begin(); it != muonCands.end(); ++it) {
    LogDebug(metname) << "Loader glbSeedRef " << (*it)->trajectory()->seedRef().isNonnull();
    if ((*it)->trackerTrajectory() )  LogDebug(metname) << " " << "tkSeedRef " << (*it)->trackerTrajectory()->seedRef().isNonnull();

    combinedTrajs.push_back((*it)->trajectory());
    if ( thePutTkTrackFlag ) trackerTrajs.push_back((*it)->trackerTrajectory());

    else {
      if ((*it)->trackerTrajectory()) delete ((*it)->trackerTrajectory());
    }
  
    // // Create the links between sta and tracker tracks
    // reco::MuonTrackLinks links;
    // links.setStandAloneTrack((*it)->muonTrack());
    // links.setTrackerTrack((*it)->trackerTrack());
    // trackLinksCollection->push_back(links);
    // delete *it;
  }
  
  // create the TrackCollection of combined Trajectories
  // FIXME: could this be done one track at a time in the previous loop?
  LogTrace(metname) << "Build combinedTracks";
  std::vector<bool> combTksVec(combinedTrajs.size(), false); 
  OrphanHandle<reco::TrackCollection> combinedTracks = loadTracks(combinedTrajs, event, combTksVec, ttopo);

  OrphanHandle<reco::TrackCollection> trackerTracks;
  std::vector<bool> trackerTksVec(trackerTrajs.size(), false); 
  if(thePutTkTrackFlag) {
    LogTrace(metname) << "Build trackerTracks: "
		      << trackerTrajs.size();
    trackerTracks = loadTracks(trackerTrajs, event, trackerTksVec, ttopo, theL2SeededTkLabel, theSmoothTkTrackFlag);
  } else {
    for (TrajectoryContainer::iterator it = trackerTrajs.begin(); it != trackerTrajs.end(); ++it) {
        if(*it) delete *it;
    }
  }

  LogTrace(metname) << "Set the final links in the MuonTrackLinks collection";

  unsigned int candposition(0), position(0), tkposition(0);
  //reco::TrackCollection::const_iterator glIt = combinedTracks->begin(), 
  //  glEnd = combinedTracks->end();

  for (CandidateContainer::const_iterator it = muonCands.begin(); it != muonCands.end(); ++it, ++candposition) {

    // The presence of the global track determines whether to fill the MuonTrackLinks or not
    // N.B. We are assuming here that the global tracks in "combinedTracks" 
    //      have the same order as the muon candidates in "muonCands"
    //      (except for possible missing tracks), which should always be the case...
    //if( glIt == glEnd ) break;
    if(combTksVec[candposition]) {
      reco::TrackRef combinedTR(combinedTracks, position++);
      //++glIt;

      // Create the links between sta and tracker tracks
      reco::MuonTrackLinks links;
      links.setStandAloneTrack((*it)->muonTrack());
      links.setTrackerTrack((*it)->trackerTrack());
      links.setGlobalTrack(combinedTR);

      if(thePutTkTrackFlag && trackerTksVec[candposition]) {
	reco::TrackRef trackerTR(trackerTracks, tkposition++);
	links.setTrackerTrack(trackerTR);
      }

      trackLinksCollection->push_back(links);
    }

    else { // if no global track, still increment the tracker-track counter when appropriate
      if(thePutTkTrackFlag && trackerTksVec[candposition]) tkposition++; 
    }

    delete *it;
  }

  if( thePutTkTrackFlag && trackerTracks.isValid() && !(combinedTracks->size() > 0 && trackerTracks->size() > 0 ) )
    LogWarning(metname)<<"The MuonTrackLinkCollection is incomplete"; 
  
  // put the MuonCollection in the event
  LogTrace(metname) << "put the MuonCollection in the event" << "\n";
  
  return event.put(trackLinksCollection);
} 

OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer& trajectories,
			    Event& event, const std::vector<std::pair<Trajectory*,reco::TrackRef> >& miniMap, Handle<reco::TrackCollection> const& trackHandle, const TrackerTopology& ttopo, const string& instance, bool reallyDoSmoothing) {
  
  const bool doSmoothing = theSmoothingStep && reallyDoSmoothing;
  
  const string metname = "Muon|RecoMuon|MuonTrackLoader|TevMuonTrackLoader";

  LogDebug(metname)<<"TeV LoadTracks instance: " << instance;
  
  // the track collectios; they will be loaded in the event  
  auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection());
  // ... and its reference into the event
  reco::TrackRefProd trackCollectionRefProd = event.getRefBeforePut<reco::TrackCollection>(instance);
    
  // Association map between GlobalMuons and TeVMuons
  auto_ptr<reco:: TrackToTrackMap> trackToTrackmap(new reco::TrackToTrackMap(trackHandle,trackCollectionRefProd ));
   
  // the track extra collection, it will be loaded in the event  
  auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );
  // ... and its reference into the event
  reco::TrackExtraRefProd trackExtraCollectionRefProd = event.getRefBeforePut<reco::TrackExtraCollection>(instance);
  
  // the rechit collection, it will be loaded in the event  
  auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );
  // ... and its reference into the event
  TrackingRecHitRefProd recHitCollectionRefProd = event.getRefBeforePut<TrackingRecHitCollection>(instance);
  
  // Collection of Trajectory
  auto_ptr<vector<Trajectory> > trajectoryCollection(new vector<Trajectory>);
  
  // don't waste any time...
  if ( trajectories.empty() ) { 
    event.put(recHitCollection,instance);
    event.put(trackExtraCollection,instance);
    if(theTrajectoryFlag) {
      event.put(trajectoryCollection,instance);

      // Association map between track and trajectory
      std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection() );
      event.put( trajTrackMap, instance );
    }
    event.put(trackToTrackmap, instance);
    return event.put(trackCollection,instance);
  }
  
  LogTrace(metname) << "Create the collection of Tracks";
  
  edm::Handle<reco::BeamSpot> beamSpot;
  event.getByToken(theBeamSpotToken,beamSpot);

  reco::TrackRef::key_type trackIndex = 0;
  //  reco::TrackRef::key_type trackUpdatedIndex = 0;
  
  reco::TrackExtraRef::key_type trackExtraIndex = 0;
  TrackingRecHitRef::key_type recHitsIndex = 0;
  
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;
  
  if(doSmoothing) {
    edm::ESHandle<TrajectorySmoother> aSmoother;
    theService->eventSetup().get<TrajectoryFitter::Record>().get(theSmootherName,aSmoother);
    theSmoother.reset(aSmoother->clone());
    edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
    try {
      std::string theTrackerRecHitBuilderName("WithAngleAndTemplate");  // to be moved to cfg in another PR
      theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
      theTrackerRecHitBuilder.product();
    } catch(...) {
      std::string theTrackerRecHitBuilderName("hltESPTTRHBWithTrackAngle");  // FIXME FIXME
      theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    }
    hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(theTrackerRecHitBuilder.product())->cloner();
    theSmoother->setHitCloner(&hitCloner);
  }

  for(TrajectoryContainer::const_iterator rawTrajectory = trajectories.begin();
      rawTrajectory != trajectories.end(); ++rawTrajectory){

    reco::TrackRef glbRef;
    std::vector<std::pair<Trajectory*,reco::TrackRef> >::const_iterator mmit;
    for(mmit = miniMap.begin();mmit!=miniMap.end();++mmit){
      if(mmit->first == *rawTrajectory) glbRef = mmit->second;
    }
    
    Trajectory &trajectory = **rawTrajectory;
    
    if(doSmoothing){
      vector<Trajectory> trajectoriesSM = theSmoother->trajectories(**rawTrajectory);
      
      if(!trajectoriesSM.empty()) {
	const edm::RefToBase<TrajectorySeed> tmpSeedRef = (**rawTrajectory).seedRef();
	trajectory = trajectoriesSM.front();
        trajectory.setSeedRef(tmpSeedRef);
      } else
	LogInfo(metname)<<"The trajectory has not been smoothed!"<<endl;
    }
    
    if(theTrajectoryFlag) {
      trajectoryCollection->push_back(trajectory);
      iTjRef++;
    }    
    
    // get the transient rechit from the trajectory
    Trajectory::RecHitContainer transHits = trajectory.recHits();
    
    if ( trajectory.direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());
    
    // build the "bare" track from the trajectory.
    // This track has the parameters defined at PCA (no update)
    pair<bool,reco::Track> resultOfTrackExtrapAtPCA = buildTrackAtPCA(trajectory, *beamSpot);
    
    // Check if the extrapolation went well    
    if(!resultOfTrackExtrapAtPCA.first) {
      //      ++trackIndex;//ADAM
      delete *rawTrajectory;
      continue;
    }
    
    // take the "bare" track at PCA
    reco::Track &track = resultOfTrackExtrapAtPCA.second;
    
    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( trajectory );
    
    // get the TrackExtraRef (persitent reference of the track extra)
    reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );
    
    // set the persistent track-extra reference to the Track
    track.setExtra(trackExtraRef);

    // Fill the map
    trackToTrackmap->insert(glbRef,
			    reco::TrackRef(trackCollectionRefProd,trackIndex++));

    // build the updated-at-vertex track, starting from the previous track
    pair<bool,reco::Track> updateResult(false,reco::Track());
            
    // Fill the track extra with the rec hit (persistent-)reference
    unsigned int nHitsAdded = 0;
    for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	 recHit != transHits.end(); ++recHit) {
	TrackingRecHit *singleHit = (**recHit).hit()->clone();
    
    std::vector<const TrackingRecHit*> hits = MuonTrackLoader::unpackHit(*singleHit);
    for (std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it) {
        if unlikely(!track.appendHitPattern(**it, ttopo)){
            break;
        }
    }
    
    if(theUpdatingAtVtx && updateResult.first){
        std::vector<const TrackingRecHit*> hits = MuonTrackLoader::unpackHit(*singleHit);
        for (std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it) {
            if unlikely(!updateResult.second.appendHitPattern(**it, ttopo)){
                break;
            }
        }
    }
    recHitCollection->push_back( singleHit );
    ++nHitsAdded;
    }
    // set the TrackingRecHitRef (persitent reference of the tracking rec hits)
    trackExtra.setHits(recHitCollectionRefProd, recHitsIndex, nHitsAdded);
    recHitsIndex += nHitsAdded;

    // fill the TrackExtraCollection
    trackExtraCollection->push_back(trackExtra);
    
    // fill the TrackCollection
    trackCollection->push_back(track);
    iTkRef++;
    LogTrace(metname) << "Debug Track being loaded pt "<< track.pt();
    
    // We don't need the original trajectory anymore.
    // It has been copied by value in the trajectoryCollection, if 
    // it is required to put it into the event.
    delete *rawTrajectory;

    if(theTrajectoryFlag) tjTkMap[iTjRef-1] = iTkRef-1;
  }
  

  
  // Put the Collections in the event
  LogTrace(metname) << "put the Collections in the event";
  event.put(recHitCollection,instance);
  event.put(trackExtraCollection,instance);

  OrphanHandle<reco::TrackCollection> returnTrackHandle;
  OrphanHandle<reco::TrackCollection> nonUpdatedHandle;
  if(theUpdatingAtVtx){
  }
  else {
    event.put(trackToTrackmap,instance);
    returnTrackHandle = event.put(trackCollection,instance);
    nonUpdatedHandle = returnTrackHandle;
  }

  if ( theTrajectoryFlag ) {
    OrphanHandle<std::vector<Trajectory> > rTrajs = event.put(trajectoryCollection,instance);

    // Association map between track and trajectory
    std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection(rTrajs, nonUpdatedHandle) );

    // Now Create traj<->tracks association map
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); 
          i != tjTkMap.end(); i++ ) {
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
                            edm::Ref<reco::TrackCollection>( nonUpdatedHandle, (*i).second ) );
    }
    event.put( trajTrackMap, instance );
  }

  return returnTrackHandle;
}


pair<bool,reco::Track> MuonTrackLoader::buildTrackAtPCA(const Trajectory& trajectory, const reco::BeamSpot &beamSpot) const {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";

  MuonPatternRecoDumper debug;
  
  // FIXME: check the prop direction
  TrajectoryStateOnSurface innerTSOS = trajectory.geometricalInnermostState();
  
  // This is needed to extrapolate the tsos at vertex
  LogTrace(metname) << "Propagate to PCA...";
  pair<bool,FreeTrajectoryState> 
    extrapolationResult = theUpdatorAtVtx->propagate(innerTSOS, beamSpot);  
  FreeTrajectoryState ftsAtVtx;
  
  if(extrapolationResult.first)
    ftsAtVtx = extrapolationResult.second;
  else{    
    if(TrackerBounds::isInside(innerTSOS.globalPosition())){
      LogInfo(metname) << "Track in the Tracker: taking the innermost state instead of the state at PCA";
      ftsAtVtx = *innerTSOS.freeState();
    }
    else{
      if ( theAllowNoVtxFlag ) {
        LogInfo(metname) << "Propagation to PCA failed, taking the innermost state instead of the state at PCA";
        ftsAtVtx = *innerTSOS.freeState();
      } else {
        LogInfo(metname) << "Stand Alone track: this track will be rejected";
        return pair<bool,reco::Track>(false,reco::Track());
      }
    }
  }
    
  LogTrace(metname) << "TSOS after the extrapolation at vtx";
  LogTrace(metname) << debug.dumpFTS(ftsAtVtx);
  
  GlobalPoint pca = ftsAtVtx.position();
  math::XYZPoint persistentPCA(pca.x(),pca.y(),pca.z());
  GlobalVector p = ftsAtVtx.momentum();
  math::XYZVector persistentMomentum(p.x(),p.y(),p.z());

  bool bon = true;
  if(fabs(theService->magneticField()->inTesla(GlobalPoint(0,0,0)).z()) < 0.01) bon=false;   
  double ndof = trajectory.ndof(bon);
  
  reco::Track track(trajectory.chiSquared(), 
		    ndof,
		    persistentPCA,
		    persistentMomentum,
		    ftsAtVtx.charge(),
		    ftsAtVtx.curvilinearError());
  
  return pair<bool,reco::Track>(true,track);
}


pair<bool,reco::Track> MuonTrackLoader::buildTrackUpdatedAtPCA(const reco::Track &track, const reco::BeamSpot &beamSpot) const {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";
  MuonPatternRecoDumper debug;
 
  // build the transient track
  reco::TransientTrack transientTrack(track,
				      &*theService->magneticField(),
				      theService->trackingGeometry());

  LogTrace(metname) << "Apply the vertex constraint";
  pair<bool,FreeTrajectoryState> updateResult = theUpdatorAtVtx->update(transientTrack,beamSpot);

  if(!updateResult.first){
    return pair<bool,reco::Track>(false,reco::Track());
  }

  LogTrace(metname) << "FTS after the vertex constraint";
  FreeTrajectoryState &ftsAtVtx = updateResult.second;

  LogTrace(metname) << debug.dumpFTS(ftsAtVtx);
  
  GlobalPoint pca = ftsAtVtx.position();
  math::XYZPoint persistentPCA(pca.x(),pca.y(),pca.z());
  GlobalVector p = ftsAtVtx.momentum();
  math::XYZVector persistentMomentum(p.x(),p.y(),p.z());
  
  reco::Track updatedTrack(track.chi2(), 
			   track.ndof(),
			   persistentPCA,
			   persistentMomentum,
			   ftsAtVtx.charge(),
			   ftsAtVtx.curvilinearError());
  
  return pair<bool,reco::Track>(true,updatedTrack);
}


reco::TrackExtra MuonTrackLoader::buildTrackExtra(const Trajectory& trajectory) const {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  // put the collection of TrackingRecHit in the event
  
  // sets the outermost and innermost TSOSs
  // FIXME: check it!
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  unsigned int innerId=0, outerId=0;
  TrajectoryMeasurement::ConstRecHitPointer outerRecHit;
  DetId outerDetId;

  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname)<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    outerRecHit =  trajectory.lastMeasurement().recHit();
    outerDetId =   trajectory.lastMeasurement().recHit()->geographicalId();
  } 
  else if (trajectory.direction() == oppositeToMomentum) {
    LogTrace(metname)<<"oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    outerRecHit =  trajectory.firstMeasurement().recHit();
    outerDetId =   trajectory.firstMeasurement().recHit()->geographicalId();
  }
  else LogError(metname)<<"Wrong propagation direction!";
  
  const GeomDet *outerDet = theService->trackingGeometry()->idToDet(outerDetId);
  GlobalPoint outerTSOSPos = outerTSOS.globalParameters().position();
  bool inside = outerDet->surface().bounds().inside(outerDet->toLocal(outerTSOSPos));

  
  GlobalPoint hitPos = (outerRecHit->isValid()) ? outerRecHit->globalPosition() :  outerTSOS.globalParameters().position() ;
  
  if(!inside) {
    LogTrace(metname)<<"The Global Muon outerMostMeasurementState is not compatible with the recHit detector! Setting outerMost postition to recHit position if recHit isValid: " << outerRecHit->isValid();
    LogTrace(metname)<<"From " << outerTSOSPos << " to " <<  hitPos;
  }
  
  
  //build the TrackExtra
  GlobalPoint v = (inside) ? outerTSOSPos : hitPos ;
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
  math::XYZVector outmom( p.x(), p.y(), p.z() );
  
  v = innerTSOS.globalParameters().position();
  p = innerTSOS.globalParameters().momentum();
  math::XYZPoint  inpos( v.x(), v.y(), v.z() );   
  math::XYZVector inmom( p.x(), p.y(), p.z() );

  reco::TrackExtra trackExtra(outpos, outmom, true, inpos, inmom, true,
                              outerTSOS.curvilinearError(), outerId,
                              innerTSOS.curvilinearError(), innerId,
			      trajectory.direction(),trajectory.seedRef());
  
  return trackExtra;
 
}

