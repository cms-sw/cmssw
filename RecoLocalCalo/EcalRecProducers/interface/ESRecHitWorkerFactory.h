#ifndef RecoLocalCalo_EcalRecProducers_ESRecHitWorkerFactory_hh
#define RecoLocalCalo_EcalRecProducers_ESRecHitWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerBaseClass.h"
typedef edmplugin::PluginFactory< ESRecHitWorkerBaseClass*(const edm::ParameterSet&) > ESRecHitWorkerFactory;

#endif
