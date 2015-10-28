/*!

This module performs the fast simulation of the reconstruction of TrajectorySeeds.
It takes as input a vector of tracking rechit combinations (FastTrackingRecHitCombination).
Seed reconstruction is only allowed within each rechit combination.

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
        
        std::vector<std::unique_ptr<TrackingRegion> > regions;
        std::unique_ptr<TrackingRegionProducer> theRegionProducer;
        const edm::EventSetup * es_;
        
        SeedingTree<TrackingLayer> _seedingTree; 

    public:

        TrajectorySeedProducer(const edm::ParameterSet& conf);
    
        virtual ~TrajectorySeedProducer()
        {
        }

        virtual void produce(edm::Event& e, const edm::EventSetup& es);

    
};

#endif
