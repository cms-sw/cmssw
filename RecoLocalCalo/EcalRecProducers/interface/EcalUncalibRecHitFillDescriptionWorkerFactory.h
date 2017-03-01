#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitFillDescriptionWorkerFactory_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitFillDescriptionWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"

typedef edmplugin::PluginFactory< EcalUncalibRecHitWorkerBaseClass*() > EcalUncalibRecHitFillDescriptionWorkerFactory;

#endif
