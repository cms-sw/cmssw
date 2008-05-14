#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripHashedDetIdESModule.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(SiStripHashedDetIdESModule);

#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripNoiseDummyCalculator.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripNoiseDummyCalculator);
