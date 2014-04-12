#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"
EDM_REGISTER_PLUGINFACTORY(ESRecHitWorkerFactory, "ESRecHitWorkerFactory");
