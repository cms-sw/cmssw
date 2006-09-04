
/** \class MuonTrackLoader
 *  Class to load the product in the event
 *
 *  $Date: 2006/08/31 18:24:18 $
 *  $Revision: 1.24 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

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
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

using namespace edm;

// constructor
MuonTrackLoader::MuonTrackLoader(std::string trackLoaderPropagatorName, const MuonServiceProxy *service): 
  thePropagatorName(trackLoaderPropagatorName),
  theService(service){}


edm::OrphanHandle<reco::TrackCollection> 
MuonTrackLoader::loadTracks(const TrajectoryContainer& trajectories,
			    edm::Event& event) {
  
  const std::string metname = "Muon|RecoMuon|MuonTrackLoader";

  // the track collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackCollection> trackCollection( new reco::TrackCollection() );
  
  // don't waste any time...
  if ( trajectories.empty() ) { 
    return event.put(trackCollection);
  }

  // the track extra collection, it will be loaded in the event  
  std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection() );
  // ... and its reference into the event
  reco::TrackExtraRefProd trackExtraCollectionRefProd = event.getRefBeforePut<reco::TrackExtraCollection>();
  
  // the rechit collection, it will be loaded in the event  
  std::auto_ptr<TrackingRecHitCollection> recHitCollection(new TrackingRecHitCollection() );
  // ... and its reference into the event
  TrackingRecHitRefProd recHitCollectionRefProd = event.getRefBeforePut<TrackingRecHitCollection>();

  LogDebug(metname) << "Create the collection of Tracks";

  reco::TrackExtraRef::key_type trackExtraIndex = 0;
  TrackingRecHitRef::key_type recHitsIndex = 0;

  for(TrajectoryContainer::const_iterator trajectory = trajectories.begin();
      trajectory != trajectories.end(); ++trajectory){
    
    // get the transient rechit from the trajectory
    Trajectory::RecHitContainer transHits = (*trajectory)->recHits();
    
    if ( (*trajectory)->direction() == oppositeToMomentum)
      reverse(transHits.begin(),transHits.end());

    // build the "bare" track from the trajectory
    reco::Track track = buildTrack( **trajectory );

    // build the "bare" track extra from the trajectory
    reco::TrackExtra trackExtra = buildTrackExtra( **trajectory );

    // get the TrackExtraRef (persitent reference of the track extra)
    reco::TrackExtraRef trackExtraRef(trackExtraCollectionRefProd, trackExtraIndex++ );
    
    // set the persistent track-extra reference to the Track
    track.setExtra(trackExtraRef);

    // Fill the track extra with the rec hit (persistent-)reference
    size_t i = 0;
    for (Trajectory::RecHitContainer::const_iterator recHit = transHits.begin();
	 recHit != transHits.end(); ++recHit) {
      if((**recHit).isValid()){
	TrackingRecHit *singleHit = (**recHit).hit()->clone();
	recHitCollection->push_back( singleHit );  
	track.setHitPattern( *singleHit, i ++ );
	// set the TrackingRecHitRef (persitent reference of the tracking rec hits)
	trackExtra.add(TrackingRecHitRef(recHitCollectionRefProd, recHitsIndex++ ));
      }
    }

    // fill the TrackExtraCollection
    trackExtraCollection->push_back(trackExtra);

    // fill the TrackCollection
    trackCollection->push_back(track);

    // clean the memory. FIXME: check this!
    Trajectory::DataContainer dataContainer = (*trajectory)->measurements();
    for (Trajectory::DataContainer::iterator datum = dataContainer.begin(); 
	 datum != dataContainer.end(); ++datum) 
      delete datum->recHit();
    
    // delete the trajectory
    delete *trajectory;
  }
  
  // Put the Collections in the event
  LogDebug(metname) << "put the Collections in the event";
  
  event.put(recHitCollection);
  event.put(trackExtraCollection);
  
  return event.put(trackCollection);
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
  TransverseImpactPointExtrapolator tipe( *theService->propagator(thePropagatorName) );
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
  unsigned int innerId=0, outerId=0;
  
  if (trajectory.direction() == alongMomentum) {
    LogDebug(metname)<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
  } 
  else if (trajectory.direction() == oppositeToMomentum) {
    LogDebug(metname)<<"oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
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

  reco::TrackExtra trackExtra(outpos, outmom, true, inpos, inmom, true,
                              outerTSOS.curvilinearError(), outerId,
                              innerTSOS.curvilinearError(), innerId);
  
  return trackExtra;
 
}
