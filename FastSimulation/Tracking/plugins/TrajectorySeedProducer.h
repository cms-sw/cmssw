#ifndef FastSimulation_Tracking_TrajectorySeedProducer2_h
#define FastSimulation_Tracking_TrajectorySeedProducer2_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FastSimulation/Tracking/interface/SeedingTree.h"
#include "FastSimulation/Tracking/interface/TrackingLayer.h"
	 //#include "DataFormats/BeamSpot/interface/BeamSpot.h"
	 //#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include <sstream>


class MagneticField;
class MagneticFieldMap;
class TrackerGeometry;
class PropagatorWithMaterial;

class TrajectorySeedProducer: public edm::stream::EDProducer <>
{
    private:
        SeedingTree<TrackingLayer> _seedingTree;


        const MagneticField* magneticField;
        const MagneticFieldMap* magneticFieldMap;
        const TrackerGeometry* trackerGeometry;
        const TrackerTopology* trackerTopology;
        
        PropagatorWithMaterial* thePropagator;

        double pTMin;
        double maxD0;
        double maxZ0;
        unsigned int minRecHits;
        edm::InputTag hitProducer;
        edm::InputTag theBeamSpot;

        bool seedCleaning;
        unsigned int absMinRecHits;
        unsigned int numberOfHits;
        
        std::string outputSeedCollectionName;


        std::vector<std::vector<TrackingLayer>> seedingLayers;
        
        math::XYZPoint beamspotPosition;

        double originRadius;
        double originHalfLength;
        double originpTMin;

        double zVertexConstraint;
        
        bool skipPVCompatibility;

        const reco::VertexCollection* vertices;
 
        // tokens
        edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
        edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken;
        edm::EDGetTokenT<edm::SimVertexContainer> simVertexToken;
        edm::EDGetTokenT<SiTrackerGSMatchedRecHit2DCollection> recHitToken;
        edm::EDGetTokenT<reco::VertexCollection> recoVertexToken;
	std::vector<edm::EDGetTokenT<std::vector<int> > > skipSimTrackIdTokens;

    public:

    TrajectorySeedProducer(const edm::ParameterSet& conf);
    
    virtual ~TrajectorySeedProducer();

    virtual void beginRun(edm::Run const& run, const edm::EventSetup & es);
    virtual void produce(edm::Event& e, const edm::EventSetup& es);

    //! method checks if a SimTrack fulfills the quality requirements.
    /*!
    \param theSimTrack the SimTrack to be tested.
    \param theSimVertex the associated SimVertex of the SimTrack.
    \return true if a track fulfills the requirements.
    */
    virtual bool passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex) const;

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

    /// Check that the seed is compatible with a track coming from within
    /// a cylinder of radius originRadius, with a decent pT.
    bool compatibleWithBeamAxis(
            const GlobalPoint& gpos1, 
            const GlobalPoint& gpos2,
            double error,
            bool forward
    ) const;

    //! method inserts hit into the tree structure at an empty position. 
    /*!
    \param trackerRecHits list of all TrackerRecHits.
    \param hitIndicesInTree hit indices which translates the tree node to the hits in \e trackerRecHits. Empty positions are identified with '-1'.
    \param node where to look for an empty position. Important for recursive tree traversing (Breadth-first). Starts with the root.
    \param trackerHit hit which is tested.
    \return pointer if this hit is inserted at a leaf which means that a seed has been found. Returns 'nullptr' otherwise.
    */
    const SeedingNode<TrackingLayer>* insertHit(
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            std::vector<int>& hitIndicesInTree,
            const SeedingNode<TrackingLayer>* node, 
            unsigned int trackerHit
    ) const;


};

#endif
