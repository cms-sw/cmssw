#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"


#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripFedCablingFakeESSource.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripFedCablingFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripHashedDetIdFakeESSource.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripHashedDetIdFakeESSource);

//---------- Empty Fake Source -----------//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateEmptyFakeESSource.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadStripFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadChannelFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadFiberFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadModuleFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripDetVOffFakeESSource);

//------------------------------------//

// Producers starting from existing tags
#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripApvGainBuilderFromTag.h"
DEFINE_FWK_MODULE(SiStripApvGainBuilderFromTag);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripNoiseNormalizedWithApvGainBuilder.h"
DEFINE_FWK_MODULE(SiStripNoiseNormalizedWithApvGainBuilder);


//---------- Phase2 Fake Source from python config -----------//

#include "CalibTracker/SiStripESProducers/plugins/fake/Phase2TrackerCablingCfgESSource.h"
DEFINE_FWK_EVENTSETUP_SOURCE(Phase2TrackerCablingCfgESSource);

