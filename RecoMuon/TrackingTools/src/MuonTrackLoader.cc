
/** \class MuonTrackLoader
 *  Class to load the product in the event
 *
 *  $Date: 2006/08/25 14:46:12 $
 *  $Revision: 1.19 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

using namespace edm;

// constructor
MuonTrackLoader::MuonTrackLoader() : thePropagator(0) {}

void MuonTrackLoader::setES(const EventSetup& setup) {
  setup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", thePropagator);
}

edm::OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer& trajectories,
			    edm::Event& event) {
  
  const std::string metname = "Muon|RecoMuon|MuonTrackLoader";
  
  // *** first loop: create the full collection of TrackingRecHit ***
  
  LogDebug(metname) << 
    "first loop: create the full collection of TrackingRecHit" << "\n";
  
  // the rechit collection, it will be loaded in the event  
  std::auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );

  // the track extra collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );

  // the track collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackCollection> trackCollection( new reco::TrackCollection() );

  // don't waste any time...
  if ( trajectories.empty() ) { 
    return event.put(trackCollection);
  }

  for (TrajectoryContainer::const_iterator trajectory = trajectories.begin();
       trajectory != trajectories.end(); ++trajectory) {
    
    // get the transient rechit from the trajectory
    Trajectory::RecHitContainer transHits = (*trajectory)->recHits();
    
    if ( (*trajectory)->direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());
    
    // fill the rechit collection
    for(Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	recHit != transHits.end(); ++recHit) {
      if((**recHit).isValid())
	recHitCollection->push_back( (**recHit).hit()->clone() );       
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
	
  for (TrajectoryContainer::const_iterator trajectory = trajectories.begin();
       trajectory != trajectories.end(); ++trajectory) {
    
    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( **trajectory );

    // get (again!) the transient rechit from the trajectory	
    Trajectory::RecHitContainer transHits = (*trajectory)->recHits();
    
    if ( (*trajectory)->direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());
    
    // Fill the track extra with the rec hit (persistent-)reference
    for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	 recHit != transHits.end(); ++recHit) {
      trackExtra.add(TrackingRecHitRef(orphanHandleRecHit,position));
      ++position;
    }
    
    // fill the TrackExtraCollection
    trackExtraCollection->push_back(trackExtra);
  }

  // put the collection of TrackExtra in the event
  LogDebug(metname) <<  "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> orphanHandleTrackExtra = event.put(trackExtraCollection);
  
  // *** third loop: create the collection of Tracks ***

  LogDebug(metname) << "third loop: create the collection of Tracks" << "\n";
  
  position = 0;

  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    // build the "bare" track from the trajectory
    reco::Track track = buildTrack( **trajectory );
    
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
    
    // fill the TrackCollection
    trackCollection->push_back(track);

    ++position;
  }
  
  // (finally!) put the TrackCollection in the event
  LogDebug(metname) << "put the TrackCollection in the event" << "\n";
  edm::OrphanHandle<reco::TrackCollection> orphanHandleTrack = event.put(trackCollection);
  
  // clean the memory. FIXME: check this!
  for (TrajectoryContainer::const_iterator trajectory = trajectories.begin();
       trajectory != trajectories.end(); ++trajectory) {
    
    Trajectory::DataContainer dataContainer = (*trajectory)->measurements();
    for (Trajectory::DataContainer::iterator datum = dataContainer.begin(); 
	 datum != dataContainer.end(); ++datum) 
      delete datum->recHit();

    // delete trajectory
    delete *trajectory;

  }

  return orphanHandleTrack;

}

edm::OrphanHandle<reco::MuonCollection> 
MuonTrackLoader::loadTracks(const CandidateContainer& muonCands,
			    edm::Event& event) {

  const std::string metname = "Muon|RecoMuon|MuonTrackLoader";
  
  // the muon collection, it will be loaded in the event
  std::auto_ptr<reco::MuonCollection> muonCollection(new reco::MuonCollection());
  // get combined Trajectories
  TrajectoryContainer combinedTrajs;
  for (CandidateContainer::const_iterator it = muonCands.begin(); it != muonCands.end(); it++) {
    combinedTrajs.push_back((*it)->trajectory());
    
    // Create the reco::muon
    reco::Muon muon;
    muon.setStandAlone((*it)->muonTrack());
    muon.setTrack((*it)->trackerTrack());
    muonCollection->push_back(muon);
    delete *it;
  }

  // create the TrackCollection of combined Trajectories
  // FIXME: could this be done one track at a time in the previous loop?
  edm::OrphanHandle<reco::TrackCollection> combinedTracks = loadTracks(combinedTrajs, event);

  reco::MuonCollection::iterator muon = muonCollection->begin();
  for ( unsigned int position = 0; position != combinedTracks->size(); position++ ) {
    reco::TrackRef combinedTR(combinedTracks, position);
    // fill the combined information.
    // FIXME: can this break in case combined info cannot be added to some tracks?
    (*muon).setCombined(combinedTR);
    muon++;
  }

  // put the MuonCollection in the event
  LogDebug(metname) << "put the MuonCollection in the event" << "\n";
  edm::OrphanHandle<reco::MuonCollection> orphanHandleMuon = event.put(muonCollection);

  return orphanHandleMuon;

}

reco::Track MuonTrackLoader::buildTrack(const Trajectory& trajectory) const {

  const std::string metname = "Muon|RecoMuon|MuonTrackLoader";

  MuonPatternRecoDumper debug;
  
  // FIXME: check the prop direction
  TrajectoryStateOnSurface innerTSOS;
  
  if (trajectory.direction() == alongMomentum) {
    LogDebug(metname)<<"alongMomentum";
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } 
  else if (trajectory.direction() == oppositeToMomentum) { 
    LogDebug(metname)<<"oppositeToMomentum";
    innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  else edm::LogError(metname)<<"Wrong propagation direction!";
  
  LogDebug(metname) << debug.dumpTSOS(innerTSOS);

  // This is needed to extrapolate the tsos at vertex
  GlobalPoint vtx(0,0,0); 
  TransverseImpactPointExtrapolator tipe(*thePropagator);
  TrajectoryStateOnSurface tscp = tipe.extrapolate(innerTSOS,vtx);
  
  if ( !tscp.isValid() ) return reco::Track(); // FIXME: how to report this?
  PerigeeConversions conv;
  double pt = 0.0;
  PerigeeTrajectoryParameters perigeeParameters = conv.ftsToPerigeeParameters(*tscp.freeState(),vtx,pt);
  PerigeeTrajectoryError perigeeError = conv.ftsToPerigeeError(*tscp.freeState());

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  float dof=0.;

  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit)
    if ((**rechit).isValid()) dof += (**rechit).dimension();
  
  float ndof = std::max(dof - 5., 0.);

  reco::Track track(trajectory.chiSquared(), 
		    ndof,
		    perigeeParameters, 
		    pt,
		    perigeeError);
  return track;

}

reco::TrackExtra MuonTrackLoader::buildTrackExtra(const Trajectory& trajectory) const {

  const std::string metname = "Muon|RecoMuon|MuonTrackLoader";

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  // put the collection of TrackingRecHit in the event
  
  // sets the outermost and innermost TSOSs
  // FIXME: check it!
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  
  if(trajectory.direction() == alongMomentum) {
    LogDebug(metname)<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } 
  else if(trajectory.direction() == oppositeToMomentum) {
    LogDebug(metname)<<"oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  else edm::LogError(metname)<<"Wrong propagation direction!";
  
  //build the TrackExtra
  GlobalPoint v = outerTSOS.globalParameters().position();
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
  math::XYZVector outmom( p.x(), p.y(), p.z() );

  v = innerTSOS.globalParameters().position();
  p = innerTSOS.globalParameters().momentum();
  math::XYZPoint  inpos( v.x(), v.y(), v.z() );   
  math::XYZVector inmom( p.x(), p.y(), p.z() );

  reco::TrackExtra trackExtra(outpos, outmom, true, inpos, inmom, true);
  
  return trackExtra;
 
}
