#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"



TrackingRecHitAlgorithm::TrackingRecHitAlgorithm()
{
}

void TrackingRecHitAlgorithm::beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup)
{
}

std::shared_ptr<TrackingRecHitProduct> TrackingRecHitAlgorithm::process(std::shared_ptr<TrackingRecHitProduct> product) const
{
    return product;
}


void TrackingRecHitAlgorithm::endEvent(edm::Event& event, edm::EventSetup& eventSetup)
{
}

TrackingRecHitAlgorithm::~TrackingRecHitAlgorithm()
{
}
