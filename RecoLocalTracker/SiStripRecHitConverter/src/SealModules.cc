#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEESProducer.h"

#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"

EVENTSETUP_DATA_REG(StripClusterParameterEstimator);
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(StripCPEESProducer)
using cms::SiStripRecHitConverter;
DEFINE_ANOTHER_FWK_MODULE(SiStripRecHitConverter)

