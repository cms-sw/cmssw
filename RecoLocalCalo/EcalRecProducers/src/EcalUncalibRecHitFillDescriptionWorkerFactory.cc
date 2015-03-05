#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
EDM_REGISTER_PLUGINFACTORY(EcalUncalibRecHitFillDescriptionWorkerFactory, "EcalUncalibRecHitFillDescriptionWorkerFactory");
