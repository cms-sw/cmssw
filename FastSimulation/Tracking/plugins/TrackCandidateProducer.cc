#include "FastSimulation/Tracking/plugins/TrackCandidateProducer.h"

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "FastSimulation/Tracking/interface/HitMaskHelper.h"

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
    : hitSplitter()
{  
  // products
  produces<TrackCandidateCollection>();
  
  // general parameters
  minNumberOfCrossedLayers = conf.getParameter<unsigned int>("MinNumberOfCrossedLayers");
  rejectOverlaps = conf.getParameter<bool>("OverlapCleaning");
  splitHits = conf.getParameter<bool>("SplitHits");

  // input tags, labels, tokens
  hitMasks_exists = conf.exists("hitMasks");
  if (hitMasks_exists){
      hitMasksToken = consumes<std::vector<bool> >(conf.getParameter<edm::InputTag>("hitMasks"));
  }

  edm::InputTag simTrackLabel = conf.getParameter<edm::InputTag>("simTracks");
  simVertexToken = consumes<edm::SimVertexContainer>(simTrackLabel);
  simTrackToken = consumes<edm::SimTrackContainer>(simTrackLabel);

  edm::InputTag seedLabel = conf.getParameter<edm::InputTag>("src");
  seedToken = consumes<edm::View<TrajectorySeed> >(seedLabel);

  edm::InputTag recHitCombinationsLabel = conf.getParameter<edm::InputTag>("recHitCombinations");
  recHitCombinationsToken = consumes<FastRecHitCombinationCollection>(recHitCombinationsLabel);
  
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
  //  Propagator* thePropagator = propagator.product()->clone();

  // get products
  edm::Handle<edm::View<TrajectorySeed> > seeds;
  e.getByToken(seedToken,seeds);

  edm::Handle<FastRecHitCombinationCollection> recHitCombinations;
  e.getByToken(recHitCombinationsToken, recHitCombinations);

  edm::Handle<edm::SimVertexContainer> simVertices;
  e.getByToken(simVertexToken,simVertices);

  edm::Handle<edm::SimTrackContainer> simTracks;
  e.getByToken(simTrackToken,simTracks);

  std::unique_ptr<HitMaskHelper> hitMaskHelper;

  // the hits to be skipped
  if (hitMasks_exists == true){
      edm::Handle<std::vector<bool> > hitMasks;
      e.getByToken(hitMasksToken,hitMasks);
      hitMaskHelper.reset(new HitMaskHelper(hitMasks.product()));
  }
  
  // output collection
  std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    

  // loop over the seeds
  for (unsigned seednr = 0; seednr < seeds->size(); ++seednr){
    
      const BasicTrajectorySeed seed = (*seeds)[seednr];
    if(seed.nHits()==0){
      edm::LogError("TrackCandidateProducer") << "empty trajectory seed in TrajectorySeedCollection" << std::endl;
      return;
    }

    // Get the combination of hits that produced the seed
    int32_t recHitCombinationIndex = -1;
    if(!trackerHitRTTI::isFast(**seed.recHits().first())){
	edm::LogError("TrackCandidateProducer") << "seed hits must be instances of FastTrackerRecHit class" << std::endl;
	exit(1);
    }
    const FastTrackerRecHitCombination & recHitCombination = (*recHitCombinations)[hitCombinationId];

    // select hits, temporarily store as TrajectorySeedHitCandidates
    std::vector<TrajectorySeedHitCandidate> recHitCandidates;
    TrajectorySeedHitCandidate recHitCandidate;
    unsigned numberOfCrossedLayers = 0;      
    for (const auto & _hit : *recHitCombination) {

	// apply hit masking
	if(hitMaskHelper 
	   && hitMaskHelper->mask(_hit.get()))
	    continue;

      recHitCandidate = TrajectorySeedHitCandidate(&_hit,trackerGeometry.product(),trackerTopology.product());
      if ( recHitCandidates.size() == 0 || !recHitCandidate.isOnTheSameLayer(recHitCandidates.back()) ) {
	++numberOfCrossedLayers;
      }

      // hit selection
      //         - always select first hit
      if(        recHitCandidates.size() == 0 ) {
	  recHitCandidates.push_back(recHitCandidate);
      }
      //         - in case of *no* verlap rejection: select all hits
      else if(   !rejectOverlaps) {
	  recHitCandidates.push_back(recHitCandidate);
      }
      //         - in case of overlap rejection: 
      //              - select hit if it is not on same layer as previous hit
      else if(   recHitCandidate.subDetId()    != recHitCandidates.back().subDetId() ||
		 recHitCandidate.layerNumber() != recHitCandidates.back().layerNumber() ) {
	  recHitCandidates.push_back(recHitCandidate);
      }
      //         - in case of overlap rejection and hit is on same layer as previous hit 
      //              - replace previous hit with current hit if it has better precision
      else if (  recHitCandidate.localError() < recHitCandidates.back().localError() ){
	  recHitCandidates.back() = recHitCandidate;

      }
    }

    // TODO: verify it makes sense to have this selection
    if ( numberOfCrossedLayers < minNumberOfCrossedLayers ) {
      continue;
    }

    // Convert TrajectorySeedHitCandidate to TrackingRecHit and split hits
    edm::OwnVector<TrackingRecHit> trackRecHits;
    for ( unsigned index = 0; index<recHitCandidates.size(); ++index ) {
	if(splitHits)
	    hitSplitter.split(*recHitCandidates[index].hit(),trackRecHits);
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
    
    // create track candidate

    //Get seedTSOS from seed PTSOD//---------------------------------------------------------------------------
    DetId seedDetId(seed.startingState().detId());
    const GeomDet* gdet = trackerGeometry->idToDet(seedDetId);
    TrajectoryStateOnSurface seedTSOS = trajectoryStateTransform::transientState(seed.startingState(), &(gdet->surface()),magneticField.product());
    //---------------------------------------------------------------------------------------------------------
    //backPropagate seedState to front recHit and get a new initial TSOS at front recHit//---------------------
    const GeomDet* initialLayer = trackerGeometry->idToDet(trackRecHits.front().geographicalId());
    //thePropagator->setPropagationDirection(oppositeToMomentum);
    const TrajectoryStateOnSurface initialTSOS = propagator->propagate(seedTSOS,initialLayer->surface()) ;
    //---------------------------------------------------------------------------------------------------------
    //Check if the TSOS is valid .
    if (!initialTSOS.isValid()) continue; 
    PTrajectoryStateOnDet PTSOD = trajectoryStateTransform::persistentState(initialTSOS,trackRecHits.front().geographicalId().rawId()); 
    TrackCandidate newTrackCandidate(trackRecHits,seed,PTSOD,edm::RefToBase<TrajectorySeed>(seeds,seednr));

    // add track candidate to output collection
    output->push_back(TrackCandidate(trackRecHits,seed,PTSOD,edm::RefToBase<TrajectorySeed>(fastSeedInfo.trajectorySeed)));
  }
  
  // Save the track candidates
  e.put(output);
}
