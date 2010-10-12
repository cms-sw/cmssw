#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitMatcherESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"


DEFINE_FWK_EVENTSETUP_MODULE(StripCPEESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(SiStripRecHitMatcherESProducer);
DEFINE_FWK_MODULE(SiStripRecHitConverter);
