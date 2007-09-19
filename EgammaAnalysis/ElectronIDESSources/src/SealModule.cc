#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "EgammaAnalysis/ElectronIDESSources/interface/ElectronLikelihoodESSource.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE (ElectronLikelihoodESSource) ;

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "EgammaAnalysis/ElectronIDAlgos/interface/ElectronLikelihood.h"

EVENTSETUP_DATA_REG( ElectronLikelihood );
