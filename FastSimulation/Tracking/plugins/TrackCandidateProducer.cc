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
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

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
#include "FastSimulation/Tracking/interface/SeedMatcher.h"

class TrackCandidateProducer : public edm::stream::EDProducer <>
{
public:
  
    explicit TrackCandidateProducer(const edm::ParameterSet& conf);
  
    void produce(edm::Event& e, const edm::EventSetup& es) override;
  
private:

    // tokens & labels
    edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken;
    edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
    edm::EDGetTokenT<std::vector<bool> > hitMasksToken;
    edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
    std::string propagatorLabel;
    
    // other data
    bool rejectOverlaps;
    bool splitHits;
    FastTrackerRecHitSplitter hitSplitter;
    double maxSeedMatchEstimator;
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
    simTrackToken = consumes<edm::SimTrackContainer>(conf.getParameter<edm::InputTag>("simTracks"));
    
    // other parameters
    maxSeedMatchEstimator = conf.getUntrackedParameter<double>("maxSeedMatchEstimator",0);
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

    edm::Handle<edm::SimTrackContainer> simTracks;
    e.getByToken(simTrackToken,simTracks);

    edm::Handle<std::vector<bool> > hitMasks;
    if (!hitMasksToken.isUninitialized())
    {
        e.getByToken(hitMasksToken,hitMasks);
    }
    
    // output collection
    std::unique_ptr<TrackCandidateCollection> output(new TrackCandidateCollection);
    
    // loop over the seeds
    for (unsigned seedIndex = 0; seedIndex < seeds->size(); ++seedIndex){

	const TrajectorySeed seed = (*seeds)[seedIndex];
	std::vector<int32_t> recHitCombinationIndices;

	// match hitless seeds to simTracks and find corresponding recHitCombination
	if(seed.nHits()==0){
	    recHitCombinationIndices = SeedMatcher::matchRecHitCombinations(
		seed,
		*recHitCombinations,
		*simTracks,
		maxSeedMatchEstimator,
		*propagator,
		*magneticField,
		*trackerGeometry);
	}
	// for normal seeds, retrieve the corresponding recHitCombination from the seed hits
	else
	{
	    int32_t icomb = fastTrackingUtilities::getRecHitCombinationIndex(seed);
	    recHitCombinationIndices.push_back(icomb);
	}

	// loop over the matched recHitCombinations
	for(auto icomb : recHitCombinationIndices)
	{
	    if(icomb < 0 || unsigned(icomb) >= recHitCombinations->size())
	    {
		throw cms::Exception("TrackCandidateProducer") << " found seed with recHitCombination out or range: " << icomb << std::endl;
	    }
	    const FastTrackerRecHitCombination & recHitCombination = (*recHitCombinations)[icomb];

	    // container for select hits
	    std::vector<const FastTrackerRecHit *> selectedRecHits;

	    // add the seed hits
	    TrajectorySeed::range seedHitRange = seed.recHits();//Hits in a seed
	    for (TrajectorySeed::const_iterator ihit = seedHitRange.first; ihit != seedHitRange.second; ++ihit)
	    {
		selectedRecHits.push_back(static_cast<const FastTrackerRecHit*>(&*ihit));
	    }

	    // prepare to skip seed hits
	    const FastTrackerRecHit * lastHitToSkip = nullptr;
	    if(!selectedRecHits.empty())
	    {
		lastHitToSkip = selectedRecHits.back();
	    }

	    // inOut or outIn tracking ?
	    bool hitsAlongMomentum = (seed.direction()== alongMomentum);

	    // add hits from combination to hit selection
	    for (unsigned hitIndex = hitsAlongMomentum ? 0 : recHitCombination.size() - 1;
		 hitIndex < recHitCombination.size();
		 hitsAlongMomentum ? ++hitIndex : --hitIndex)
	    {

		const FastTrackerRecHit * selectedRecHit = recHitCombination[hitIndex].get();

		// skip seed hits
		if(lastHitToSkip)
		{
		    if(lastHitToSkip->sameId(selectedRecHit))
		    {
			lastHitToSkip=nullptr;
		    }
		    continue;
		}

		// apply hit masking
		if(hitMasks.isValid() && fastTrackingUtilities::hitIsMasked(selectedRecHit,*hitMasks))
		{
		    continue;
		}


		//  if overlap rejection is not switched on, accept all hits
		//  always accept the first hit
		//  also accept a hit if it is not on the layer of the previous hit
		if( !  rejectOverlaps
		    || selectedRecHits.empty()
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
	    edm::OwnVector<TrackingRecHit> hitsForTrackCandidate;
	    for ( unsigned index = 0; index<selectedRecHits.size(); ++index )
	    {
		if(splitHits)
		{
		    // add split hits to splitSelectedRecHits
		    hitSplitter.split(*selectedRecHits[index],hitsForTrackCandidate,hitsAlongMomentum);
		}
		else
		{
		    hitsForTrackCandidate.push_back(selectedRecHits[index]->clone());
		}
	    }

	    // set the recHitCombinationIndex
	    fastTrackingUtilities::setRecHitCombinationIndex(hitsForTrackCandidate,icomb);

	    // create track candidate state
	    //   1. get seed state (defined on the surface of the most outer hit)
	    DetId seedDetId(seed.startingState().detId());
	    const GeomDet* gdet = trackerGeometry->idToDet(seedDetId);
	    TrajectoryStateOnSurface seedTSOS = trajectoryStateTransform::transientState(seed.startingState(), &(gdet->surface()),magneticField.product());
	    //   2. backPropagate the seedState to the surfuce of the most inner hit
	    const GeomDet* initialLayer = trackerGeometry->idToDet(hitsForTrackCandidate.front().geographicalId());
	    const TrajectoryStateOnSurface initialTSOS = propagator->propagate(seedTSOS,initialLayer->surface()) ;
	    //   3. check validity and transform
	    if (!initialTSOS.isValid()) continue;
	    PTrajectoryStateOnDet PTSOD = trajectoryStateTransform::persistentState(initialTSOS,hitsForTrackCandidate.front().geographicalId().rawId());

	    // add track candidate to output collection
	    output->push_back(TrackCandidate(hitsForTrackCandidate,seed,PTSOD,edm::RefToBase<TrajectorySeed>(seeds,seedIndex)));
	}
    }
  
    // Save the track candidates
    e.put(std::move(output));
    
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackCandidateProducer);
