#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripQualityFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripFedCablingFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripFedCablingFakeESSource);

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripHashedDetIdFakeESSource.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripHashedDetIdFakeESSource);

 //------------ NEW Template -------------------

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateFakeESSource.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripNoises, SiStripNoisesRcd, SiStripNoisesGenerator > SiStripNoisesFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripNoisesFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripPedestals, SiStripPedestalsRcd, SiStripPedestalsGenerator > SiStripPedestalsFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripPedestalsFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripThresholdGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripThreshold, SiStripThresholdRcd, SiStripThresholdGenerator > SiStripThresholdFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripThresholdFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripApvGain, SiStripApvGainRcd, SiStripApvGainGenerator > SiStripApvGainFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripApvGainFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripLorentzAngle, SiStripLorentzAngleRcd, SiStripLorentzAngleGenerator > SiStripLorentzAngleFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripLorentzAngleFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripBadModuleGenerator.h"
typedef SiStripTemplateFakeESSource< SiStripBadStrip, SiStripBadModuleRcd, SiStripBadModuleGenerator > SiStripBadModuleConfigurableFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadModuleConfigurableFakeESSource);


//---------- Empty Fake Source -----------//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripTemplateEmptyFakeESSource.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadStripFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadChannelFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadFiberFakeESSource);

typedef SiStripTemplateEmptyFakeESSource<SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripBadModuleFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripModuleHV,SiStripModuleHVRcd> SiStripModuleHVFakeESSource;  
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripModuleHVFakeESSource);  
 
typedef SiStripTemplateEmptyFakeESSource<SiStripModuleHV,SiStripModuleLVRcd> SiStripModuleLVFakeESSource;  
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripModuleLVFakeESSource);

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
typedef SiStripTemplateEmptyFakeESSource<SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffFakeESSource;
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripDetVOffFakeESSource);

//------------------------------------//
