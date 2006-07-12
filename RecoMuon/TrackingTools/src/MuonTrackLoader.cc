
/** \class MuonTrackLoader
 *  Class to load the product in the event
 *
 *  $Date: 2006/07/10 16:35:08 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

// FIXME!!!
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

edm::OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer &trajectories,
			    edm::Event& event){
  
  std::string metname = "Muon|RecoMuon|MuonTrackLoader";
  
  // *** first loop: create the full collection of TrackingRecHit ***
  
  LogDebug(metname) << 
    "first loop: create the full collection of TrackingRecHit" << "\n";
  
  // the rechit collection, it will be loaded in the event  
  std::auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );

  // the track extra collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );

  // the track collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackCollection> trackCollection( new reco::TrackCollection() );

  // Don't waste any time...
  if( !trajectories.size() ){ 
    return event.put(trackCollection);
  }

  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    // get the transient rechit from the trajectory
    const Trajectory::RecHitContainer transHits = trajectory->recHits();

    // fill the rechit collection
    for(Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	recHit != transHits.end(); ++recHit){
      if(recHit->isValid())
	recHitCollection->push_back( recHit->hit()->clone() );       
    }
  }
  
  // put the collection of TrackingRecHit in the event
  LogDebug(metname) << 
    "put the collection of TrackingRecHit in the event" << "\n";
  
  edm::OrphanHandle<TrackingRecHitCollection> orphanHandleRecHit = event.put( recHitCollection );
  
  // *** second loop: create the collection of TrackExtra ***

  LogDebug(metname) << 
    "second loop: create the collection of TrackExtra" << "\n";

  int position = 0;
	
  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( *trajectory );

    // get (again!) the transient rechit from the trajectory	
    const Trajectory::RecHitContainer transHits = trajectory->recHits();

    // Fill the track extra with the rec hit (persistent-)reference
    for(Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	recHit != transHits.end(); ++recHit){
      
      trackExtra.add(TrackingRecHitRef(orphanHandleRecHit,position));
      ++position;
    }
    
    // fill the TrackExtraCollection
    trackExtraCollection ->push_back(trackExtra);
  }

  //put the collection of TrackExtra in the event
  LogDebug(metname) <<  "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> orphanHandleTrackExtra = event.put(trackExtraCollection);
  
  // *** third loop: create the collection of Tracks ***

  LogDebug(metname) << "third loop: create the collection of Tracks" << "\n";
  
  position = 0;

  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    // build the "bare" track from the trajectory
    reco::Track track = buildTrack( *trajectory );
    
    // get the TrackExtraRef (persitent reference of the track extra)
    reco::TrackExtraRef trackExtraRef(orphanHandleTrackExtra,position);
    
    // set the persistent track-extra reference to the Track
    track.setExtra(trackExtraRef);

    // Hit Pattern
    //     TrackingRecHitRefVector hitlist;
    //     for (unsigned int i=0; i<trackExtraRef->recHitsSize(); i++) {
    // 	    hitlist.push_back(trackExtraRef->recHit(i));
    //     }
    
    //     track.setHitPattern(hitlist);
    
    //fill the TrackCollection
    trackCollection->push_back(track);
    
    ++position;
  }
  
  // (finally!) put the TrackCollection in the event
  LogDebug(metname) << "put the TrackCollection in the event" << "\n";
  edm::OrphanHandle<reco::TrackCollection> orphanHandleTrack = event.put(trackCollection);
  
  // clean the memory. FIXME: check this!
  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    Trajectory::DataContainer dataContainer = trajectory->measurements();
    for (Trajectory::DataContainer::iterator datum = dataContainer.begin(); 
	 datum != dataContainer.end(); ++datum) 
      delete datum->recHit();
  }  
  return orphanHandleTrack;
}

reco::Track MuonTrackLoader::buildTrack (const Trajectory& trajectory) const {

  MuonPatternRecoDumper debug;
  
  // FIXME: check the prop direction
  TrajectoryStateOnSurface innerTSOS;
  
  if (trajectory.direction() == alongMomentum) {
    LogDebug("RecoMuon")<<"alongMomentum";
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } 
  else if (trajectory.direction() == oppositeToMomentum) { 
    LogDebug("RecoMuon")<<"oppositeToMentum";
    innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  else edm::LogError("RecoMuon")<<"Wrong propagation direction!";
  
  std::string metname = "RecoMuon";
  debug.dumpTSOS(innerTSOS,metname);

  // This is needed to extrapolate the tsos at vertex
  // FIXME: check it!
  TSCPBuilderNoMaterial tscpBuilder;
  TrajectoryStateClosestToPoint tscp;
  
  try{
    tscp = tscpBuilder( innerTSOS,GlobalPoint(0,0,0) );//FIXME Correct?
  }
  catch(const TrajectoryStateException &er){
    edm::LogWarning("RecoMuon") << "caught TrajectoryStateException: "<< er.what() << std::endl;
    return reco::Track(); 
  }
  catch(const std::exception& er){
    edm::LogWarning("RecoMuon") << "caught std::exception: " << er.what() << std::endl;
    return reco::Track(); 
  }
  catch(...){
    edm::LogWarning("RecoMuon") << "Funny error" << std::endl;
    return reco::Track(); 
  }
  
  reco::perigee::Parameters param = tscp.perigeeParameters();
  reco::perigee::Covariance covar = tscp.perigeeError();


  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  int dof=0;

  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit)
    if ((*rechit).isValid()) dof += (*rechit).dimension();
  
  unsigned short int ndof = std::max(dof - 5, 0);
  
  reco::Track track(trajectory.chiSquared(), 
		    trajectory.foundHits(),//FIXME to be fixed in Trajectory.h
		    0, //FIXME no corresponding method in trajectory.h
		    trajectory.lostHits(),//FIXME to be fixed in Trajectory.h
		    ndof, param, covar);
  return track;
}


reco::TrackExtra MuonTrackLoader::buildTrackExtra(const Trajectory& trajectory) const{

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  // put the collection of TrackingRecHit in the event
  
  // sets the outermost and innermost TSOSs
  // FIXME: check it!
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  
  if(trajectory.direction() == alongMomentum) {
    LogDebug("RecoMuon")<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } 
  else if(trajectory.direction() == oppositeToMomentum) {
      LogDebug("RecoMuon")<<"oppositeToMentum";
      outerTSOS = trajectory.firstMeasurement().updatedState();
      innerTSOS = trajectory.lastMeasurement().updatedState();
    }
  else edm::LogError("RecoMuon")<<"Wrong propagation direction!";
  
  //build the TrackExtra
  GlobalPoint v = outerTSOS.globalParameters().position();
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZVector outmom( p.x(), p.y(), p.z() );
  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   

  reco::TrackExtra trackExtra(outpos, outmom, true);
  
  return trackExtra;
 
}


