#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerFactory_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
typedef edmplugin::PluginFactory< EcalUncalibRecHitWorkerBaseClass*(const edm::ParameterSet&) > EcalUncalibRecHitWorkerFactory;

#endif
