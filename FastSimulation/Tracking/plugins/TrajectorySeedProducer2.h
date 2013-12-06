#ifndef FastSimulation_Tracking_TrajectorySeedProducer2_h
#define FastSimulation_Tracking_TrajectorySeedProducer2_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>
#include <string>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrajectorySeedProducer2 : public TrajectorySeedProducer
{
 public:
  
  explicit TrajectorySeedProducer2(const edm::ParameterSet& conf);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

  virtual bool passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId);

  virtual bool passTrackerRecHitQualityCuts(std::vector<TrackerRecHit>& trackerRecHits, std::vector<unsigned int> previousHits, TrackerRecHit& currentHit, unsigned int trackingAlgorithmId);

  virtual int iterateHits(
	SiTrackerGSMatchedRecHit2DCollection::const_iterator start,
	SiTrackerGSMatchedRecHit2DCollection::range range,
	std::vector<std::vector<unsigned int>> hitNumbers,
	std::vector<TrackerRecHit>& trackerRecHits,
	unsigned int trackingAlgorithmId,
	std::vector<unsigned int>& seedHitNumbers
  );

};

#endif
