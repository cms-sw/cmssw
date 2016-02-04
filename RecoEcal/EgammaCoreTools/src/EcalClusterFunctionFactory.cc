#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"


#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
EDM_REGISTER_PLUGINFACTORY(EcalClusterFunctionFactory, "EcalClusterFunctionFactory");
