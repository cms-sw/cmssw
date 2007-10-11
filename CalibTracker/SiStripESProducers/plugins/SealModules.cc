#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/SiStripQualityFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/SiStripQualityESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);

#include "CalibTracker/SiStripESProducers/plugins/SiStripGainESProducer.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);

#include "CalibTracker/SiStripESProducers/plugins/SiStripGainFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripGainFakeESSource);






