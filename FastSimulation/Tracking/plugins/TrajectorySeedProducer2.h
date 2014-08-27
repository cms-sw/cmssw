#ifndef FastSimulation_Tracking_TrajectorySeedProducer2_h
#define FastSimulation_Tracking_TrajectorySeedProducer2_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "FastSimulation/Tracking/plugins/SeedingTree.h"
#include "FastSimulation/Tracking/interface/TrackingLayer.h"

#include <vector>
#include <sstream>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrajectorySeedProducer2 : public TrajectorySeedProducer
{
 private:
    SeedingTree<TrackingLayer> _seedingTree;
 public:

	virtual ~TrajectorySeedProducer2()
	{
	}
  
  explicit TrajectorySeedProducer2(const edm::ParameterSet& conf);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

    //! method checks if a SimTrack fulfills the requirements of the current seeding algorithm iteration.
    /*!
    \param theSimTrack the SimTrack to be tested.
    \param theSimVertex the associated SimVertex of the SimTrack.
    \param trackingAlgorithmId id of the seeding algorithm iteration (e.g. "initial step", etc.).
    \return true if a track fulfills the requirements.
    */
   virtual bool passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId) const;
  
     //! method checks if a TrackerRecHit fulfills the requirements of the current seeding algorithm iteration.
    /*!
    \param trackerRecHits list of all TrackerRecHits.
    \param previousHits list of indexes of hits which already got accepted before.
    \param currentHit the current hit which needs to pass the criteria in addition to those in \e previousHits.
    \param trackingAlgorithmId id of the seeding algorithm iteration (e.g. "initial step", etc.).
    \return true if a hit fulfills the requirements.
    */
  inline bool passHitTuplesCuts(const SeedingNode<TrackingLayer>& seedingNode,
        const std::vector<TrackerRecHit>& trackerRecHits,
		const std::vector<int>& hitIndicesInTree,
		const TrackerRecHit& currentTrackerHit,
		unsigned int trackingAlgorithmId
 ) const
 {
    switch (seedingNode.getDepth())
    {
        case 0:
        {
            return true;
            /* example for 1 hits
                const TrackerRecHit& hit1 = currentTrackerHit;
                return pass1HitsCuts(hit1,trackingAlgorithmId);
            */
        }
            
        case 1:
        {
            const SeedingNode<TrackingLayer>* parentNode = &seedingNode;
            parentNode = parentNode->getParent();
            const TrackerRecHit& hit1 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
            const TrackerRecHit& hit2 = currentTrackerHit;
            
            return pass2HitsCuts(hit1,hit2,trackingAlgorithmId);
        }
        case 2:
        {
            return true;
            /* example for 3 hits
                const SeedingNode<LayerSpec>* parentNode = &seedingNode;
                parentNode = parentNode->getParent();
                const TrackerRecHit& hit2 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                parentNode = parentNode->getParent();
                const TrackerRecHit& hit1 = trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                const TrackerRecHit& hit3 = currentTrackerHit;
                return pass3HitsCuts(hit1,hit2,hit3,trackingAlgorithmId);
            */
        }
    }
    return true;
 }
    
  bool pass2HitsCuts(const TrackerRecHit& hit1, const TrackerRecHit& hit2, unsigned int trackingAlgorithmId) const;
  
  
  virtual std::vector<unsigned int> iterateHits(
	unsigned int start,
	const std::vector<TrackerRecHit>& trackerRecHits,
	std::vector<int> hitIndicesInTree,
	bool processSkippedHits,
	unsigned int trackingAlgorithmId
  ) const;
  
  inline bool isHitOnLayer(const TrackerRecHit& trackerRecHit, const TrackingLayer& layer) const
  {
    return layer==trackerRecHit.getTrackingLayer();
  }
  
const SeedingNode<TrackingLayer>* insertHit(
    const std::vector<TrackerRecHit>& trackerRecHits,
    std::vector<int>& hitIndicesInTree,
    const SeedingNode<TrackingLayer>* node, unsigned int trackerHit,const unsigned int trackingAlgorithmId
) const;

};

#endif
