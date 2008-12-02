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

//---------- Bad Strips Empty Fake Source -----------//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateEmptyFakeESSource.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadStripFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripBadChannelRcd.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadChannelFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripBadFiberRcd.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadFiberFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripBadModuleRcd.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadModuleFakeESSource);

//------------------------------------//
