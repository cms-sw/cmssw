// system
#include <memory>
#include <vector>
#include <map>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data format
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"

// geometry / magnetic field / propagation
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

// fastsim
#include "FastSimulation/Tracking/interface/TrackingLayer.h"
#include "FastSimulation/Tracking/interface/FastTrackingUtilities.h"
#include "FastSimulation/Tracking/interface/FastTrackerRecHitSplitter.h"

class TrackCandidateProducer : public edm::stream::EDProducer <>
{
public:
  
    explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
    virtual void produce(edm::Event& e, const edm::EventSetup& es) override;
  
private:

    // tokens & labels
    edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;
    edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
    edm::EDGetTokenT<std::vector<bool> > hitMasksToken;
    std::string propagatorLabel;
    
    // other data
    bool rejectOverlaps;
    bool splitHits;
    FastTrackerRecHitSplitter hitSplitter;
  
};

TrackCandidateProducer::TrackCandidateProducer(const edm::ParameterSet& conf)
    : hitSplitter()
{  
    // produces
    produces<TrackCandidateCollection>();

    // consumes
    if (conf.exists("hitMasks")){
	hitMasksToken = consumes<std::vector<bool> >(conf.getParameter<edm::InputTag>("hitMasks"));
    }
    seedToken = consumes<edm::View<TrajectorySeed> >(conf.getParameter<edm::InputTag>("src"));
    recHitCombinationsToken = consumes<FastTrackerRecHitCombinationCollection>(conf.getParameter<edm::InputTag>("recHitCombinations"));
  
    // other parameters
    rejectOverlaps = conf.getParameter<bool>("OverlapCleaning");
    splitHits = conf.getParameter<bool>("SplitHits");
    propagatorLabel = conf.getParameter<std::string>("propagator");
}
  
void 
TrackCandidateProducer::produce(edm::Event& e, const edm::EventSetup& es) {        

    // get records
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

    edm::Handle<FastTrackerRecHitCombinationCollection> recHitCombinations;
    e.getByToken(recHitCombinationsToken, recHitCombinations);

    const std::vector<bool> * hitMasks = 0;
    if (!hitMasksToken.isUninitialized()){
	edm::Handle<std::vector<bool> > hitMasksHandle;
	e.getByToken(hitMasksToken,hitMasksHandle);
	hitMasks = &(*hitMasksHandle);
    }
    
    // output collection
    std::unique_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);    
    
    // loop over the seeds
    for (unsigned seedIndex = 0; seedIndex < seeds->size(); ++seedIndex){

	const TrajectorySeed seed = (*seeds)[seedIndex];

	// hit-less seeds are not allowed
	if(seed.nHits()==0){
	    edm::LogError("TrackCandidateProducer") << "found hit-less seed in in TrajectorySeedCollection: skip" << std::endl;
	    continue;
	}
	
	// container for select hits
	std::vector<const FastTrackerRecHit *> selectedRecHits;

	// add the hits from the seed
	auto seedHitRange = seed.recHits();
	for(auto seedHit = seedHitRange.first;seedHit != seedHitRange.second;seedHit++)
	{
	    if(!trackerHitRTTI::isFast(*seedHit))
	    {
		throw cms::Exception("TrackCandidateProducer") << "found seed with non-FastSim hit. skip event..." << std::endl;
	    }
	    selectedRecHits.push_back(static_cast<const FastTrackerRecHit *>(&(*seedHit)));
	}

	// store the id number of the outer seed hit
	int outerSeedHitId = -1;
	if(selectedRecHits.size() > 0)
	{
	    outerSeedHitId = ((FastTrackerRecHit*)selectedRecHits.back())->id();
	}

	// Get the combination of hits that produced the seed
	int32_t icomb = fastTrackingUtilities::getRecHitCombinationIndex(seed);
	if(icomb < 0 || unsigned(icomb) >= recHitCombinations->size()){
	    throw cms::Exception("TrackCandidateProducer") << " found seed with recHitCombination out or range: " << icomb << std::endl;
	}
	const FastTrackerRecHitCombination & recHitCombination = (*recHitCombinations)[icomb];

	// add hits from combination to hit selection
	for (const auto & _hit : recHitCombination) {
	    
	    const FastTrackerRecHit * selectedRecHit = _hit.get();

	    // skip until the outer seed hit is passed
	    if(outerSeedHitId >=0)
	    {
		if(outerSeedHitId == selectedRecHit->id())
		{
		    outerSeedHitId = -1;
		}
		continue;
	    }

	    // apply hit masking
	    if(hitMasks && fastTrackingUtilities::hitIsMasked(selectedRecHit,hitMasks))
	    {
		continue;
	    }


	    //  if overlap rejection is not switched on, accept all hits
	    //  always accept the first hit
	    //  also accept a hit if it is not on the layer of the previous hit
	    if( !  rejectOverlaps
		|| selectedRecHits.size() == 0 
		|| ( TrackingLayer::createFromDetId(selectedRecHits.back()->geographicalId(),*trackerTopology.product())
		     != TrackingLayer::createFromDetId(selectedRecHit->geographicalId(),*trackerTopology.product())))
	    {
		selectedRecHits.push_back(selectedRecHit);
	    }
	    //  else:
	    //    overlap rejection is switched on
	    //    the hit is on the same layer as the previous hit
	    //  accept the one with smallest error
	    else if ( fastTrackingUtilities::hitLocalError(selectedRecHit) 
		      < fastTrackingUtilities::hitLocalError(selectedRecHits.back()) )
	    {
		selectedRecHits.back() = selectedRecHit;
	    }
	}

	// split hits / store copies for the track candidate
	edm::OwnVector<TrackingRecHit> splitSelectedRecHits;
	for ( unsigned index = 0; index<selectedRecHits.size(); ++index ) 
	{
	    if(splitHits)
	    {
		// add split hits to splitSelectedRecHits
		hitSplitter.split(*selectedRecHits[index],splitSelectedRecHits);
	    }
	    else 
	    {
		splitSelectedRecHits.push_back(selectedRecHits[index]->clone());
	    }
	}
	
	// order hits along the seed direction
	// (happens for muon-seeded tracks)
	if (seed.direction()==oppositeToMomentum){
	    splitSelectedRecHits.reverse();
	}

	// set the recHitCombinationIndex
	fastTrackingUtilities::setRecHitCombinationIndex(splitSelectedRecHits,icomb);

	// create track candidate state
	//   1. get seed state (defined on the surface of the most outer hit)
	DetId seedDetId(seed.startingState().detId());
	const GeomDet* gdet = trackerGeometry->idToDet(seedDetId);
	TrajectoryStateOnSurface seedTSOS = trajectoryStateTransform::transientState(seed.startingState(), &(gdet->surface()),magneticField.product());
	//   2. backPropagate the seedState to the surfuce of the most inner hit
	const GeomDet* initialLayer = trackerGeometry->idToDet(splitSelectedRecHits.front().geographicalId());
	const TrajectoryStateOnSurface initialTSOS = propagator->propagate(seedTSOS,initialLayer->surface()) ;
	//   3. check validity and transform
	if (!initialTSOS.isValid()) continue; 
	PTrajectoryStateOnDet PTSOD = trajectoryStateTransform::persistentState(initialTSOS,splitSelectedRecHits.front().geographicalId().rawId()); 

	// add track candidate to output collection
	output->push_back(TrackCandidate(splitSelectedRecHits,seed,PTSOD,edm::RefToBase<TrajectorySeed>(seeds,seedIndex)));
    }
  
    // Save the track candidates
    e.put(std::move(output));
    
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackCandidateProducer);
