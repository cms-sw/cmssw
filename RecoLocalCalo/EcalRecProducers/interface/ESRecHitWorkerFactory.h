#ifndef RecoLocalCalo_EcalRecProducers_ESRecHitWorkerFactory_hh
#define RecoLocalCalo_EcalRecProducers_ESRecHitWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerBaseClass.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
typedef edmplugin::PluginFactory<ESRecHitWorkerBaseClass*(const edm::ParameterSet&, edm::ConsumesCollector)>
    ESRecHitWorkerFactory;

#endif
