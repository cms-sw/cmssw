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

class TrajectorySeedProducer:
    public edm::stream::EDProducer<>
{
    private:
        SeedingTree<TrackingLayer> _seedingTree;

	std::unique_ptr<SeedCreator> seedCreator;

        std::vector<std::vector<TrackingLayer>> seedingLayers;
	
	std::string measurementTrackerLabel;
	const MeasurementTracker * measurementTracker;
	
        // tokens
        edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
	edm::EDGetTokenT<std::vector<bool> > hitMasksToken;        

    public:

    TrajectorySeedProducer(const edm::ParameterSet& conf);
    
    virtual ~TrajectorySeedProducer()
    {
    }

    virtual void produce(edm::Event& e, const edm::EventSetup& es);

    //! method checks if a SimTrack fulfills the quality requirements.
    /*!
    \param theSimTrack the SimTrack to be tested.
    \param theSimVertex the associated SimVertex of the SimTrack.
    \return true if a track fulfills the requirements.
    */
    //virtual bool passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex) const;

    //! method checks if a TrajectorySeedHitCandidate fulfills the quality requirements.
    /*!
    \param seedingNode tree node at which the hit will be inserted. 
    \param trackerRecHits list of all TrackerRecHits.
    \param hitIndicesInTree hit indices which translates the tree node to the hits in \e trackerRecHits.
    \param currentTrackerHit hit which is tested.
    \return true if a hit fulfills the requirements.
    */
    inline bool passHitTuplesCuts(
            const SeedingNode<TrackingLayer>& seedingNode,
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            const std::vector<int>& hitIndicesInTree,
            const TrajectorySeedHitCandidate& currentTrackerHit
        ) const
    {
        switch (seedingNode.getDepth())
        {
            case 0:
            {
                return true;
                /* example for 1 hits
                const TrajectorySeedHitCandidate& hit1 = currentTrackerHit;
                return pass1HitsCuts(hit1,trackingAlgorithmId);
                */
            }

            case 1:
            {
                const SeedingNode<TrackingLayer>* parentNode = &seedingNode;
                parentNode = parentNode->getParent();
                const TrajectorySeedHitCandidate& hit1 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                const TrajectorySeedHitCandidate& hit2 = currentTrackerHit;

                return pass2HitsCuts(hit1,hit2);
            }
            case 2:
            {
                return true;
                /* example for 3 hits
                const SeedingNode<LayerSpec>* parentNode = &seedingNode;
                parentNode = parentNode->getParent();
                const TrajectorySeedHitCandidate& hit2 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                parentNode = parentNode->getParent();
                const TrajectorySeedHitCandidate& hit1 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                const TrajectorySeedHitCandidate& hit3 = currentTrackerHit;
                return pass3HitsCuts(hit1,hit2,hit3,trackingAlgorithmId);
                */
            }
        }
        return true;
    }

    bool pass2HitsCuts(const TrajectorySeedHitCandidate& hit1, const TrajectorySeedHitCandidate& hit2) const;

    //! method tries to insert all hits into the tree structure.
    /*!
    \param start index where to begin insertion. Important for recursion. 
    \param trackerRecHits list of all TrackerRecHits.
    \param hitIndicesInTree hit indices which translates the tree node to the hits in \e trackerRecHits.
    \param currentTrackerHit hit which is tested.
    \return list of hit indices which form a found seed. Returns empty list if no seed was found.
    */
    virtual std::vector<unsigned int> iterateHits(
            unsigned int start,
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            std::vector<int> hitIndicesInTree,
            bool processSkippedHits
        ) const;

    inline bool isHitOnLayer(const TrajectorySeedHitCandidate& trackerRecHit, const TrackingLayer& layer) const
    {
        return layer==trackerRecHit.getTrackingLayer();
    }

    const SeedingNode<TrackingLayer>* insertHit(
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            std::vector<int>& hitIndicesInTree,
            const SeedingNode<TrackingLayer>* node, 
            unsigned int trackerHit
    ) const;
    
    std::vector<std::unique_ptr<TrackingRegion> > regions;
    std::unique_ptr<TrackingRegionProducer> theRegionProducer;
    const edm::EventSetup * es_;


};

#endif
