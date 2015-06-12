#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
//#include "FastSimulation/Tracking/interface/TrackerRecHitSplit.h"

#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"

#include <vector>
#include <map>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//Propagator withMaterial
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf)
{  
  // products
  produces<TrackCandidateCollection>();
  
  // general parameters
  minNumberOfCrossedLayers = conf.getParameter<unsigned int>("MinNumberOfCrossedLayers");
  rejectOverlaps = conf.getParameter<bool>("OverlapCleaning");
  splitHits = conf.getParameter<bool>("SplitHits");

  // input tags, labels, tokens
  edm::InputTag simTrackLabel = conf.getParameter<edm::InputTag>("simTracks");
  simVertexToken = consumes<edm::SimVertexContainer>(simTrackLabel);
  simTrackToken = consumes<edm::SimTrackContainer>(simTrackLabel);

  edm::InputTag seedLabel = conf.getParameter<edm::InputTag>("src");
  seedToken = consumes<edm::View<TrajectorySeed> >(seedLabel);

  edm::InputTag recHitLabel = conf.getParameter<edm::InputTag>("recHits");
  recHitToken = consumes<SiTrackerGSMatchedRecHit2DCollection>(recHitLabel);
  
  propagatorLabel = conf.getParameter<std::string>("propagator");
}
  
void 
TrackCandidateProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

  // get services
  edm::ESHandle<MagneticField>          magneticField;
  es.get<IdealMagneticFieldRecord>().get(magneticField);

  edm::ESHandle<TrackerGeometry>        trackerGeometry;
  es.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

  edm::ESHandle<TrackerTopology>        trackerTopology;
  es.get<TrackerTopologyRcd>().get(trackerTopology);

  edm::ESHandle<Propagator>             propagator;
  es.get<TrackingComponentsRecord>().get(propagatorLabel,propagator);
    
  // get products
  edm::Handle<edm::View<TrajectorySeed> > seeds;
  e.getByToken(seedToken,seeds);

  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> recHits;
  e.getByToken(recHitToken, recHits);

  edm::Handle<edm::SimVertexContainer> simVertices;
  e.getByToken(simVertexToken,simVertices);

  edm::Handle<edm::SimTrackContainer> simTracks;
  e.getByToken(simTrackToken,simTracks);
  
  // output collection
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    

  // loop over the seeds
  for (unsigned seednr = 0; seednr < seeds->size(); ++seednr){
    
    const BasicTrajectorySeed seed = seeds->at(seednr);
    if(seed.nHits()==0){
      edm::LogError("TrackCandidateProducer") << "empty trajectory seed in TrajectorySeedCollection" << std::endl;
      return;
    }

    // Get all the rechits associated to this track
    int simTrackId =  ((const SiTrackerGSMatchedRecHit2D*) (&*(seed.recHits().first)))->simtrackId();
    SiTrackerGSMatchedRecHit2DCollection::range recHitRange = recHits->get(simTrackId);
    SiTrackerGSMatchedRecHit2DCollection::const_iterator recHitIter = recHitRange.first;
    SiTrackerGSMatchedRecHit2DCollection::const_iterator recHitEnd  = recHitRange.second;

    // Count number of crossed layers, apply overlap rejection
    std::vector<TrajectorySeedHitCandidate> recHitCandidates;
    TrajectorySeedHitCandidate recHitCandidate;
    unsigned numberOfCrossedLayers = 0;      
    for ( ; recHitIter != recHitEnd; ++recHitIter) {
      recHitCandidate = TrajectorySeedHitCandidate(&(*recHitIter),trackerGeometry.product(),trackerTopology.product());
      if ( recHitCandidates.size() == 0 || !recHitCandidate.isOnTheSameLayer(recHitCandidates.back()) ) {
	++numberOfCrossedLayers;
      }

      if( recHitCandidates.size() == 0 ||                                                // add the first seeding hit in any case
	  !rejectOverlaps ||                                                             // without overlap rejection:   add each hit
	  recHitCandidate.subDetId()    != recHitCandidates.back().subDetId() ||         // with overlap rejection:      only add if hits are not on the same layer
	  recHitCandidate.layerNumber() != recHitCandidates.back().layerNumber() ){
	recHitCandidates.push_back(recHitCandidate);
      }
      else if ( recHitCandidate.localError() < recHitCandidates.back().localError() ){
	recHitCandidates.back() = recHitCandidate;
      }
    }
    if ( numberOfCrossedLayers < minNumberOfCrossedLayers ) {
      continue;
    }

    // Convert TrajectorySeedHitCandidate to TrackingRecHit and split hits
    edm::OwnVector<TrackingRecHit> trackRecHits;
    for ( unsigned index = 0; index<recHitCandidates.size(); ++index ) {
      if(splitHits && recHitCandidates[index].matchedHit()->isMatched()){
	trackRecHits.push_back(recHitCandidates[index].matchedHit()->monoHit()->clone());
	trackRecHits.push_back(recHitCandidates[index].matchedHit()->stereoHit()->clone());
      }
      else {
	trackRecHits.push_back(recHitCandidates[index].hit()->clone());
      }
    }
    // reverse order if needed
    // when is this relevant? perhaps for the cases when track finding goes backwards?
    if (seed.direction()==oppositeToMomentum){
      LogDebug("FastTracking")<<"reversing the order of the hits";
      std::reverse(recHitCandidates.begin(),recHitCandidates.end());
    }
    
    // initial track candidate parameters parameters
    int vertexIndex = simTracks->at(simTrackId).vertIndex();
    GlobalPoint  position(simVertices->at(vertexIndex).position().x(),
			  simVertices->at(vertexIndex).position().y(),
			  simVertices->at(vertexIndex).position().z());
    GlobalVector momentum( simTracks->at(simTrackId).momentum().x() , 
			   simTracks->at(simTrackId).momentum().y() , 
			   simTracks->at(simTrackId).momentum().z() );
    float        charge   = simTracks->at(simTrackId).charge();
    GlobalTrajectoryParameters initialParams(position,momentum,(int)charge,magneticField.product());
    AlgebraicSymMatrix55 errorMatrix= AlgebraicMatrixID();    
    CurvilinearTrajectoryError initialError(errorMatrix);
    FreeTrajectoryState initialFTS(initialParams, initialError);      

    // create track candidate
    const GeomDet* initialLayer = trackerGeometry->idToDet(trackRecHits.front().geographicalId());
    const TrajectoryStateOnSurface initialTSOS = propagator->propagate(initialFTS,initialLayer->surface()) ;
    if (!initialTSOS.isValid()) continue; 
    PTrajectoryStateOnDet PTSOD = trajectoryStateTransform::persistentState(initialTSOS,trackRecHits.front().geographicalId().rawId()); 
    TrackCandidate newTrackCandidate(trackRecHits,seed,PTSOD,edm::RefToBase<TrajectorySeed>(seeds,seednr));

    // add track candidate to output collection
    output->push_back(newTrackCandidate);
    
  }
  
  // Save the track candidates
  e.put(output);
}
