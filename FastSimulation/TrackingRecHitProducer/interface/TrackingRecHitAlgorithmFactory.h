#ifndef FastSimulation_Tracking_RecHitAlgorithmFactory_H
#define FastSimulation_Tracking_RecHitAlgorithmFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include <string>

namespace edm
{
    class ParameterSet;
    class ConsumesCollector;
}


typedef edmplugin::PluginFactory<TrackingRecHitAlgorithm*(const std::string&, const edm::ParameterSet&, edm::ConsumesCollector&)> TrackingRecHitAlgorithmFactory;


#endif
