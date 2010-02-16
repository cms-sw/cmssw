#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitMatcherESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(StripCPEESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripRecHitMatcherESProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripRecHitConverter);
TYPELOOKUP_DATA_REG(SiStripRecHitMatcher);

