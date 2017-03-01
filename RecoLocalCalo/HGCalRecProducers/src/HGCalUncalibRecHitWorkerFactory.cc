#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerFactory.h"
EDM_REGISTER_PLUGINFACTORY(HGCalUncalibRecHitWorkerFactory, "HGCalUncalibRecHitWorkerFactory");
