#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
EDM_REGISTER_PLUGINFACTORY(HGCalRecHitWorkerFactory, "HGCalRecHitWorkerFactory");
