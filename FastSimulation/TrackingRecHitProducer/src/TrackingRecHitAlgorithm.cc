#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrackingRecHitAlgorithm::TrackingRecHitAlgorithm(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector
):
    _name(name),
    _selectionString(config.getParameter<std::string>("select")),
    _trackerTopology(nullptr),
    _trackerGeometry(nullptr),
    _misalignedTrackerGeometry(nullptr),
    _randomEngine(nullptr)
{
}

const TrackerTopology& TrackingRecHitAlgorithm::getTrackerTopology() const
{
    if (!_trackerTopology)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": TrackerTopology not defined";
    }
    return *_trackerTopology;
}

const TrackerGeometry& TrackingRecHitAlgorithm::getTrackerGeometry() const
{
    if (!_trackerGeometry)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": TrackerGeometry not defined";
    }
    return *_trackerGeometry;
}

const TrackerGeometry& TrackingRecHitAlgorithm::getMisalignedGeometry() const
{
    if (!_misalignedTrackerGeometry)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": MisalignedGeometry not defined";
    }
    return *_misalignedTrackerGeometry;
}

const RandomEngineAndDistribution& TrackingRecHitAlgorithm::getRandomEngine() const
{
    if (!_randomEngine)
    {
        throw cms::Exception("TrackingRecHitAlgorithm ") << _name <<": RandomEngineAndDistribution not defined";
    }
    return *_randomEngine;
}

void TrackingRecHitAlgorithm::beginStream(const edm::StreamID& id)
{
  _randomEngine = std::make_shared<RandomEngineAndDistribution>(id);
}


void TrackingRecHitAlgorithm::beginRun(edm::Run const& run, const edm::EventSetup& eventSetup,
				       const SiPixelTemplateDBObject * pixelTemplateDBObjectPtr,
				       std::vector< SiPixelTemplateStore > & tempStoreRef )
{
  // The default is to do nothing.
}


void TrackingRecHitAlgorithm::beginEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<TrackerGeometry> misalignedGeometryHandle;

    eventSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
    eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    eventSetup.get<TrackerDigiGeometryRecord>().get("MisAligned",misalignedGeometryHandle);

    _trackerTopology = trackerTopologyHandle.product();
    _trackerGeometry = trackerGeometryHandle.product();
    _misalignedTrackerGeometry = misalignedGeometryHandle.product();

}

TrackingRecHitProductPtr TrackingRecHitAlgorithm::process(TrackingRecHitProductPtr product) const
{
    return product;
}

void TrackingRecHitAlgorithm::endEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
    //set these to 0 -> ensures that beginEvent needs to be executed before accessing these pointers again
    _trackerGeometry=nullptr;
    _trackerTopology=nullptr;
    _misalignedTrackerGeometry=nullptr;
}

void TrackingRecHitAlgorithm::endStream()
{
    _randomEngine.reset();
}

TrackingRecHitAlgorithm::~TrackingRecHitAlgorithm()
{
}
