#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerFactory_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerFactory_hh

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"
typedef edmplugin::PluginFactory<HGCalRecHitWorkerBaseClass*(const edm::ParameterSet&, edm::ConsumesCollector)>
    HGCalRecHitWorkerFactory;

#endif
