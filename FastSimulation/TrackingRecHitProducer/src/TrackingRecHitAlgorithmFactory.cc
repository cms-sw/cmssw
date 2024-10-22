#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"

EDM_REGISTER_PLUGINFACTORY(TrackingRecHitAlgorithmFactory, "TrackingRecHitAlgorithmFactory");
