#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithmFactory_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithmFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"

#include <string>

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm

typedef edmplugin::PluginFactory<TrackingRecHitAlgorithm*(
    const std::string&, const edm::ParameterSet&, edm::ConsumesCollector&)>
    TrackingRecHitAlgorithmFactory;

#endif
