#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityConfigurableFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityConfigurableFakeESSource);

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

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadStripFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadStripFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadChannelFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadChannelFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadFiberFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadFiberFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripBadModuleFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadModuleFakeESSource);
