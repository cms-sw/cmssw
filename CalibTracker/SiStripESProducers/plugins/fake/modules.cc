#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripFedCablingFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripFedCablingFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripHashedDetIdFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripHashedDetIdFakeESSource);

 //------------ NEW Template -------------------

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateFakeESSource.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripNoises, SiStripNoisesRcd, SiStripNoisesGenerator > SiStripNoisesFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripNoisesFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripPedestals, SiStripPedestalsRcd, SiStripPedestalsGenerator > SiStripPedestalsFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripPedestalsFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripThreshold, SiStripThresholdRcd, SiStripThresholdGenerator > SiStripThresholdFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripThresholdFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripApvGain, SiStripApvGainRcd, SiStripApvGainGenerator > SiStripGainFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripGainFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripLorentzAngle, SiStripLorentzAngleRcd, SiStripLorentzAngleGenerator > SiStripLorentzAngleFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripLorentzAngleFakeESSource);

#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBadStripGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripBadStrip, SiStripBadStripRcd, SiStripBadStripGenerator > SiStripBadStripConfigurableFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadStripConfigurableFakeESSource);


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
