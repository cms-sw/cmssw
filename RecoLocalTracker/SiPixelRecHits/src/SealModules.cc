// SealModules: declarations of our Framework components to the Framework


//--- Our components which we want Framework to know about:
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEInitialESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEParmErrorESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelRecHitConverter.h"



//--- The header files for the Framework infrastructure (macros etc):
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"



//--- Now use the Framework macros to set it all up:
//
EVENTSETUP_RECORD_REG(TrackerCPERecord);
using cms::SiPixelRecHitConverter;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelCPEInitialESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(PixelCPEParmErrorESProducer);
DEFINE_ANOTHER_FWK_MODULE(SiPixelRecHitConverter);
