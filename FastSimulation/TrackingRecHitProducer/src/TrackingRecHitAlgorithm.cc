#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Utilities/interface/Exception.h"



TrackingRecHitAlgorithm::TrackingRecHitAlgorithm(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector
):
    _name(name),
    _selectionString(config.getParameter<std::string>("select")),
    _trackerTopology(nullptr),
    _trackerGeometry(nullptr),
    _randomEngine(nullptr)
{
    
}

const TrackerTopology* TrackingRecHitAlgorithm::getTrackerTopology() const
{
    if (!_trackerTopology)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": TrackerTopology not defined";
    }
    return _trackerTopology;
}

const TrackerGeometry* TrackingRecHitAlgorithm::getTrackerGeometry() const
{
    if (!_trackerGeometry)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": TrackerGeometry not defined";
    }
    return _trackerGeometry;
}

const RandomEngineAndDistribution* TrackingRecHitAlgorithm::getRandomEngine() const
{
    if (!_randomEngine)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": RandomEngineAndDistribution not defined";
    }
    return _randomEngine.get();
}

void TrackingRecHitAlgorithm::beginStream(const edm::StreamID& id)
{
    _randomEngine = std::make_shared<RandomEngineAndDistribution>(id);
}

void TrackingRecHitAlgorithm::beginEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;
    eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    eventSetup.get<IdealGeometryRecord>().get(trackerTopologyHandle);
    _trackerGeometry = trackerGeometryHandle.product();
    _trackerTopology = trackerTopologyHandle.product();
}

std::shared_ptr<TrackingRecHitProduct> TrackingRecHitAlgorithm::process(std::shared_ptr<TrackingRecHitProduct> product) const
{
    return product;
}

void TrackingRecHitAlgorithm::endEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
    //set these to 0 -> ensures that beginEvent needs to be executed before accessing these pointers again
    _trackerGeometry=nullptr;
    _trackerTopology=nullptr;
}

void TrackingRecHitAlgorithm::endStream()
{
}

TrackingRecHitAlgorithm::~TrackingRecHitAlgorithm()
{
}
