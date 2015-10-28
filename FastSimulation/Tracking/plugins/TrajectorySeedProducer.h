/*!

This module performs the reconstruction of TrajectorySeeds in FastSim.

The main input data of this modules is a vector of tracking rechit combinations.
Each rechit combination is itself a vector of tracking rechits.
The combinations are considered separately, one by one,
and for each combination, TrajectorySeedProducer attempts to reconstruct a seed from the hits inside that combination. 
Inside a combination, hits are considered in the given order.

Optionally, TrajectorySeedProducer can be configured to mask a given list of hits from the seed reconstruction.



*/

#ifndef FastSimulation_Tracking_TrajectorySeedProducer_h
#define FastSimulation_Tracking_TrajectorySeedProducer_h

// system
#include <memory>
#include <vector>
#include <sstream>
#include <string>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

// data formats 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

// reco track classes
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

// geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// fastsim
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "FastSimulation/Tracking/interface/SeedingTree.h"
#include "FastSimulation/Tracking/interface/TrackingLayer.h"
#include "FastSimulation/Tracking/interface/FastTrackingUtilities.h"
#include "FastSimulation/Tracking/interface/SeedFinder.h"

class TrajectorySeedProducer:
    public edm::stream::EDProducer<>
{
    private:


        std::unique_ptr<SeedCreator> seedCreator;

        std::vector<std::vector<TrackingLayer>> seedingLayers;

        std::string measurementTrackerLabel;
        const MeasurementTracker * measurementTracker;

        // tokens
        edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
        edm::EDGetTokenT<std::vector<bool> > hitMasksToken;       
        
        std::unique_ptr<TrackingRegionProducer> theRegionProducer;
        SeedingTree<TrackingLayer> _seedingTree; 

    public:

        TrajectorySeedProducer(const edm::ParameterSet& conf);
    
        virtual void produce(edm::Event& e, const edm::EventSetup& es);

    
};

#endif
