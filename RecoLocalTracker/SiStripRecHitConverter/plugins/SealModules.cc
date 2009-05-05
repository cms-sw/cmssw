#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitMatcherESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(StripCPEESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripRecHitMatcherESProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripRecHitConverter);
EVENTSETUP_DATA_REG(SiStripRecHitMatcher);

