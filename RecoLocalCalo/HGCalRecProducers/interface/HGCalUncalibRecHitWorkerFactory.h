#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerFactory_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerBaseClass.h"
typedef edmplugin::PluginFactory<HGCalUncalibRecHitWorkerBaseClass*(const edm::ParameterSet&, edm::ConsumesCollector)>
    HGCalUncalibRecHitWorkerFactory;

#endif
