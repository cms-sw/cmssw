#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"



TrackingRecHitAlgorithm::TrackingRecHitAlgorithm(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector
):
    _trackerTopology(nullptr)
{
    _selectionString=config.getParameter<std::string>("select");
}

void TrackingRecHitAlgorithm::beginEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
}

std::shared_ptr<TrackingRecHitProduct> TrackingRecHitAlgorithm::process(std::shared_ptr<TrackingRecHitProduct> product) const
{
    return product;
}


void TrackingRecHitAlgorithm::endEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
}

TrackingRecHitAlgorithm::~TrackingRecHitAlgorithm()
{
}
