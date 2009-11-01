#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_SEAL_MODULE();

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//-----------------------------------------------------------------------------------------

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/SiStripFedCablingManipulator.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripFedCablingManipulator);
//-----------------------------------------------------------------------------------------

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/DummyCondDBWriter.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
typedef DummyCondDBWriter<SiStripFedCabling,SiStripFedCabling,SiStripFedCablingRcd> SiStripFedCablingDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripFedCablingDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
typedef DummyCondDBWriter<SiStripPedestals,SiStripPedestals,SiStripPedestalsRcd> SiStripPedestalsDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripPedestalsDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef DummyCondDBWriter<SiStripNoises,SiStripNoises,SiStripNoisesRcd> SiStripNoisesDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripNoisesDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef DummyCondDBWriter<SiStripApvGain,SiStripApvGain,SiStripApvGainRcd> SiStripApvGainDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripApvGainDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
typedef DummyCondDBWriter<SiStripLorentzAngle,SiStripLorentzAngle,SiStripLorentzAngleRcd> SiStripLorentzAngleDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripLorentzAngleDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
typedef DummyCondDBWriter<SiStripThreshold,SiStripThreshold,SiStripThresholdRcd> SiStripThresholdDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadModuleDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadFiberDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadChannelDummyDBWriter);

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
typedef DummyCondDBWriter<SiStripQuality,SiStripBadStrip,SiStripQualityRcd> SiStripBadStripFromQualityDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripFromQualityDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
typedef DummyCondDBWriter<SiStripModuleHV,SiStripModuleHV,SiStripModuleHVRcd> SiStripModuleHVDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripModuleHVDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
typedef DummyCondDBWriter<SiStripDetVOff,SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripDetVOffDummyDBWriter);


//---------------------------------------------------------------------------------------------------------------
// Dummy printers

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/DummyCondObjPrinter.h"

typedef DummyCondObjPrinter<SiStripNoises,SiStripNoisesRcd> SiStripNoises_DecModeDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripNoises_DecModeDummyPrinter);

typedef DummyCondObjPrinter<SiStripNoises,SiStripNoisesRcd> SiStripNoises_PeakModeDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripNoises_PeakModeDummyPrinter);

typedef DummyCondObjPrinter<SiStripThreshold,SiStripThresholdRcd> SiStripThresholdDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdDummyPrinter);

typedef DummyCondObjPrinter<SiStripThreshold,SiStripThresholdRcd> SiStripClusterThresholdDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripClusterThresholdDummyPrinter);

typedef DummyCondObjPrinter<SiStripLorentzAngle,SiStripLorentzAngleRcd> SiStripLorentzAngleDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripLorentzAngleDummyPrinter);

typedef DummyCondObjPrinter<SiStripPedestals,SiStripPedestalsRcd> SiStripPedestalsDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripPedestalsDummyPrinter);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef DummyCondObjPrinter<SiStripApvGain,SiStripApvGainRcd> SiStripApvGainDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripApvGainDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
typedef DummyCondObjPrinter<SiStripGain,SiStripGainRcd> SiStripGainDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripGainDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
typedef DummyCondObjPrinter<SiStripGain,SiStripGainSimRcd> SiStripGainSimDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripGainSimDummyPrinter);

typedef DummyCondObjPrinter<SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripDetVOffDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
typedef DummyCondObjPrinter<SiStripDetCabling,SiStripDetCablingRcd> SiStripDetCablingDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripDetCablingDummyPrinter);

typedef DummyCondObjPrinter<SiStripFedCabling,SiStripFedCablingRcd> SiStripFedCablingDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripFedCablingDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadModuleDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadFiberDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelDummyPrinter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadChannelDummyPrinter);
