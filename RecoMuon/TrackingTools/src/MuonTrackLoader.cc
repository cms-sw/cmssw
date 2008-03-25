
/** \class MuonTrackLoader
 *  Class to load the product in the event
 *

 *  $Date: 2007/07/30 13:16:40 $
 *  $Revision: 1.51 $

 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

using namespace edm;
using namespace std;

// constructor
MuonTrackLoader::MuonTrackLoader(ParameterSet &parameterSet, const MuonServiceProxy *service): 
  theService(service){

  // option to do or not the smoothing step.
  // the trajectories which are passed to the track loader are supposed to be non-smoothed
  theSmoothingStep = parameterSet.getParameter<bool>("DoSmoothing");
  if(theSmoothingStep)
    theSmootherName = parameterSet.getParameter<string>("Smoother");  
  
  // update at vertex
  theUpdatingAtVtx = parameterSet.getParameter<bool>("VertexConstraint");
  
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
			    Event& event, const string& instance, bool reallyDoSmoothing) {

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
  auto_ptr<reco:: TrackToTrackMap> trackToTrackmap(new reco::TrackToTrackMap);

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
    if(theTrajectoryFlag) event.put(trajectoryCollection,instance);
    if(theUpdatingAtVtx){
      event.put(trackToTrackmap);
      event.put(updatedAtVtxTrackCollection,instance+"UpdatedAtVtx");
    }
    return event.put(trackCollection,instance);
  }
  
  LogTrace(metname) << "Create the collection of Tracks";

  reco::TrackRef::key_type trackIndex = 0;
  reco::TrackRef::key_type trackUpdatedIndex = 0;

  reco::TrackExtraRef::key_type trackExtraIndex = 0;
  TrackingRecHitRef::key_type recHitsIndex = 0;

   if(doSmoothing)
     theService->eventSetup().get<TrackingComponentsRecord>().get(theSmootherName,theSmoother);
  

  for(TrajectoryContainer::const_iterator rawTrajectory = trajectories.begin();
      rawTrajectory != trajectories.end(); ++rawTrajectory){
    
    Trajectory &trajectory = **rawTrajectory;

      if(doSmoothing){
	vector<Trajectory> trajectoriesSM = theSmoother->trajectories(**rawTrajectory);
      
      if(!trajectoriesSM.empty())
	trajectory = trajectoriesSM.front();
      else
	LogWarning(metname)<<"The trajectory has not been smoothed!"<<endl;
    }

    if(theTrajectoryFlag) trajectoryCollection->push_back(trajectory);
    
    // get the transient rechit from the trajectory
    Trajectory::RecHitContainer transHits = trajectory.recHits();

    if ( trajectory.direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());
    
    // build the "bare" track from the trajectory.
    // This track has the parameters defined at PCA (no update)
    pair<bool,reco::Track> resultOfTrackExtrapAtPCA = buildTrackAtPCA(trajectory);
    
    // Check if the extrapolation went well    
    if(!resultOfTrackExtrapAtPCA.first) continue;

    // take the "bare" track at PCA
    reco::Track &track = resultOfTrackExtrapAtPCA.second;
       
    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( trajectory );

    // get the TrackExtraRef (persitent reference of the track extra)
    reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );

    // set the persistent track-extra reference to the Track
    track.setExtra(trackExtraRef);

    // build the updated-at-vertex track, starting from the previous track
    reco::Track updatedTrack;

    if(theUpdatingAtVtx){
      // build the "bare" track UPDATED at vtx
      updatedTrack = buildTrackUpdatedAtPCA(track);

      // set the persistent track-extra reference to the Track
      updatedTrack.setExtra(trackExtraRef);

      // Fill the map
      trackToTrackmap->insert(reco::TrackRef(trackCollectionRefProd,trackIndex++),
			      reco::TrackRef(trackUpdatedCollectionRefProd,trackUpdatedIndex++));
    }
    
    // Fill the track extra with the rec hit (persistent-)reference
    size_t i = 0;
    for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	 recHit != transHits.end(); ++recHit) {
      if((**recHit).isValid()){
	TrackingRecHit *singleHit = (**recHit).hit()->clone();
	track.setHitPattern( *singleHit, i ++ );
	if(theUpdatingAtVtx) updatedTrack.setHitPattern( *singleHit, i ++ );
	recHitCollection->push_back( singleHit );  
	// set the TrackingRecHitRef (persitent reference of the tracking rec hits)
	trackExtra.add(TrackingRecHitRef(recHitCollectionRefProd, recHitsIndex++ ));
      }
    }

    // fill the TrackExtraCollection
    trackExtraCollection->push_back(trackExtra);

    // fill the TrackCollection
    trackCollection->push_back(track);
    LogTrace(metname) << "Debug Track being loaded pt "<<  track.pt();
    // fill the TrackCollection updated at vtx
    if(theUpdatingAtVtx) updatedAtVtxTrackCollection->push_back(updatedTrack);

    // We don't need the original trajectory anymore.
    // It has been copied by value in the trajectoryCollection, if 
    // it is required to put it into the event.
     delete *rawTrajectory;
  }
  
  // Put the Collections in the event
  LogTrace(metname) << "put the Collections in the event";
  
  event.put(recHitCollection,instance);
  event.put(trackExtraCollection,instance);
  if ( theTrajectoryFlag ) event.put(trajectoryCollection,instance);

  if(theUpdatingAtVtx){
    event.put(trackCollection,instance);
    event.put(trackToTrackmap);
    return  event.put(updatedAtVtxTrackCollection,instance+"UpdatedAtVtx");
  }
  else
    return event.put(trackCollection,instance);
}

OrphanHandle<reco::MuonTrackLinksCollection> 
MuonTrackLoader::loadTracks(const CandidateContainer& muonCands,
			    Event& event) {

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
      loadTracks(trackerTrajs, event, theL2SeededTkLabel, theSmoothTkTrackFlag);
    }

    return event.put(trackLinksCollection);
  }
  
  // get combined Trajectories
  TrajectoryContainer combinedTrajs;
  TrajectoryContainer trackerTrajs;
  for (CandidateContainer::const_iterator it = muonCands.begin(); it != muonCands.end(); it++) {
    combinedTrajs.push_back((*it)->trajectory());
    trackerTrajs.push_back((*it)->trackerTrajectory());
  
    // Create the links between sta and tracker tracks
    reco::MuonTrackLinks links;
    links.setStandAloneTrack((*it)->muonTrack());
    links.setTrackerTrack((*it)->trackerTrack());
    trackLinksCollection->push_back(links);
    delete *it;
  }
  
  // create the TrackCollection of combined Trajectories
  // FIXME: could this be done one track at a time in the previous loop?
  OrphanHandle<reco::TrackCollection> combinedTracks = loadTracks(combinedTrajs, event);

  OrphanHandle<reco::TrackCollection> trackerTracks;
  if(thePutTkTrackFlag) trackerTracks = loadTracks(trackerTrajs, event, theL2SeededTkLabel, theSmoothTkTrackFlag);

  
  reco::MuonTrackLinksCollection::iterator links = trackLinksCollection->begin();
  for ( unsigned int position = 0; position != combinedTracks->size(); ++position, ++links) {
    reco::TrackRef combinedTR(combinedTracks, position);

    reco::TrackRef trackerTR;
    if(thePutTkTrackFlag) trackerTR = reco::TrackRef(trackerTracks, position);

    // fill the combined information.
    links->setGlobalTrack(combinedTR);
    if(thePutTkTrackFlag) links->setTrackerTrack(trackerTR);
  }
  
  // put the MuonCollection in the event
  LogTrace(metname) << "put the MuonCollection in the event" << "\n";

  return event.put(trackLinksCollection);
}

pair<bool,reco::Track> MuonTrackLoader::buildTrackAtPCA(const Trajectory& trajectory) const {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";

  MuonPatternRecoDumper debug;
  
  // FIXME: check the prop direction
  TrajectoryStateOnSurface innerTSOS;
  
  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname)<<"alongMomentum";
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } 
  else if (trajectory.direction() == oppositeToMomentum) { 
    LogTrace(metname)<<"oppositeToMomentum";
    innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  else LogError(metname)<<"Wrong propagation direction!";
  
  LogTrace(metname) << "TSOS before the extrapolation at PCA";
  LogTrace(metname) << debug.dumpTSOS(innerTSOS);
  LogTrace(metname) << innerTSOS.freeState()->parameters();
  LogTrace(metname) << "Cartesian Errors";
  LogTrace(metname) << innerTSOS.freeState()->cartesianError().matrix();
  LogTrace(metname) << "Curvilinear Errors";
  LogTrace(metname) << innerTSOS.freeState()->curvilinearError().matrix();


  // This is needed to extrapolate the tsos at vertex
  LogTrace(metname) << "Propagate to PCA...";
  pair<bool,FreeTrajectoryState> 
    extrapolationResult = theUpdatorAtVtx->propagate(innerTSOS);  
  FreeTrajectoryState ftsAtVtx;
  
  if(extrapolationResult.first)
    ftsAtVtx = extrapolationResult.second;
  else{    
    if(TrackerBounds::isInside(innerTSOS.globalPosition())){
      LogWarning(metname) << "Track in the Tracker: taking the innermost state instead of the state at PCA";
      ftsAtVtx = *innerTSOS.freeState();
    }
    else{
      if ( theAllowNoVtxFlag ) {
        LogWarning(metname) << "Propagation to PCA failed, taking the innermost state instead of the state at PCA";
        ftsAtVtx = *innerTSOS.freeState();
      } else {
        LogWarning(metname) << "Stand Alone track: this track will be rejected";
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
  
  double ndof = computeNDOF(trajectory);
  
  reco::Track track(trajectory.chiSquared(), 
		    ndof,
		    persistentPCA,
		    persistentMomentum,
		    ftsAtVtx.charge(),
		    ftsAtVtx.curvilinearError());
  
  return pair<bool,reco::Track>(true,track);
}


reco::Track MuonTrackLoader::buildTrackUpdatedAtPCA(const reco::Track &track) const {

  const string metname = "Muon|RecoMuon|MuonTrackLoader";
  MuonPatternRecoDumper debug;
 
  // build the transient track
  reco::TransientTrack transientTrack(track,
				      &*theService->magneticField(),
				      theService->trackingGeometry());

  LogTrace(metname) << "Apply the vertex constraint";
  pair<bool,FreeTrajectoryState> updateResult = theUpdatorAtVtx->update(transientTrack);

  if(!updateResult.first){
    return reco::Track();
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
  
  return updatedTrack;
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
  
  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname)<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
  } 
  else if (trajectory.direction() == oppositeToMomentum) {
    LogTrace(metname)<<"oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
  }
  else LogError(metname)<<"Wrong propagation direction!";
  
  //build the TrackExtra
  GlobalPoint v = outerTSOS.globalParameters().position();
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
			      trajectory.direction());
  
  return trackExtra;
 
}


double MuonTrackLoader::computeNDOF(const Trajectory& trajectory) const {
  
  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  double ndof = 0.;
  
  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit)
    if ((*rechit)->isValid()) ndof += (*rechit)->dimension();
  
  // FIXME! in case of Boff is dof - 4
  return max(ndof - 5., 0.);
}
