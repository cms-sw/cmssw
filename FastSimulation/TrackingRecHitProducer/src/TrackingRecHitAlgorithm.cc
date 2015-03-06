#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"



TrackingRecHitAlgorithm::TrackingRecHitAlgorithm()
{
}

void TrackingRecHitAlgorithm::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup)
{
}

std::vector<SiTrackerGSRecHit2D> TrackingRecHitAlgorithm::processDetId(const DetId& detId, const std::vector<const PSimHit*>& simHits) const
{
    return std::vector<SiTrackerGSRecHit2D>();
}

void TrackingRecHitAlgorithm::endEvent(edm::Event& event, edm::EventSetup& eventSetup)
{
}

TrackingRecHitAlgorithm::~TrackingRecHitAlgorithm()
{
}
