//#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "EgammaAnalysis/ElectronIDESSources/plugins/ElectronLikelihoodESSource.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE (ElectronLikelihoodESSource) ;

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"

EVENTSETUP_DATA_REG( ElectronLikelihood );
