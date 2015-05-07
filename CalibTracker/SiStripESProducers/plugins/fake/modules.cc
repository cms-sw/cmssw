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

 //------------ NEW Template -------------------

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateFakeESSource.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripPedestals, SiStripPedestalsRcd, SiStripPedestalsGenerator > SiStripPedestalsFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripPedestalsFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripThreshold, SiStripThresholdRcd, SiStripThresholdGenerator > SiStripThresholdFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripThresholdFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripApvGain, SiStripApvGainRcd, SiStripApvGainGenerator > SiStripApvGainFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripApvGainFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLatencyGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripLatency, SiStripLatencyRcd, SiStripLatencyGenerator > SiStripLatencyFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripLatencyFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBaseDelayGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripBaseDelay, SiStripBaseDelayRcd, SiStripBaseDelayGenerator > SiStripBaseDelayFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBaseDelayFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripConfObjectGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripConfObject, SiStripConfObjectRcd, SiStripConfObjectGenerator > SiStripConfObjectFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripConfObjectFakeESSource);

//------------ Dependent Sources -------------------
#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateDepFakeESSource.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
typedef SiStripTemplateDepFakeESSource< SiStripNoises, SiStripNoisesDepRcd, SiStripNoisesGenerator,IdealGeometryRecord,TrackerTopology  > SiStripNoisesFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripNoisesFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBadModuleGenerator.h"
typedef SiStripTemplateDepFakeESSource< SiStripBadStrip, SiStripBadModuleDepRcd, SiStripBadModuleGenerator,IdealGeometryRecord,TrackerTopology > SiStripBadModuleConfigurableFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadModuleConfigurableFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
typedef SiStripTemplateDepFakeESSource< SiStripLorentzAngle, SiStripLorentzAngleDepRcd, SiStripLorentzAngleGenerator,IdealGeometryRecord,TrackerTopology > SiStripLorentzAngleFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripLorentzAngleFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBackPlaneCorrectionGenerator.h"
typedef SiStripTemplateDepFakeESSource< SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionDepRcd, SiStripBackPlaneCorrectionGenerator,IdealGeometryRecord,TrackerTopology > SiStripBackPlaneCorrectionFakeESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBackPlaneCorrectionFakeESSource);


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

