#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripGainFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripGainFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripFedCablingFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripFedCablingFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripHashedDetIdFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripHashedDetIdFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripNoiseFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripNoiseFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripPedestalsFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripPedestalsFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripThresholdFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripThresholdFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripThresholdFakeOnDB.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdFakeOnDB);

