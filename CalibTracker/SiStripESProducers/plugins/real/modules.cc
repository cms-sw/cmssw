#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);

#include "CalibTracker/SiStripESProducers/plugins/real/SiStripGainESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);

