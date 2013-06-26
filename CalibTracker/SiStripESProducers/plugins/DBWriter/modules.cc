#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"



#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//-----------------------------------------------------------------------------------------

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/SiStripFedCablingManipulator.h"
DEFINE_FWK_MODULE(SiStripFedCablingManipulator);
//-----------------------------------------------------------------------------------------

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/DummyCondDBWriter.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
typedef DummyCondDBWriter<SiStripFedCabling,SiStripFedCabling,SiStripFedCablingRcd> SiStripFedCablingDummyDBWriter;
DEFINE_FWK_MODULE(SiStripFedCablingDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
typedef DummyCondDBWriter<SiStripPedestals,SiStripPedestals,SiStripPedestalsRcd> SiStripPedestalsDummyDBWriter;
DEFINE_FWK_MODULE(SiStripPedestalsDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef DummyCondDBWriter<SiStripNoises,SiStripNoises,SiStripNoisesRcd> SiStripNoisesDummyDBWriter;
DEFINE_FWK_MODULE(SiStripNoisesDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef DummyCondDBWriter<SiStripApvGain,SiStripApvGain,SiStripApvGainRcd> SiStripApvGainDummyDBWriter;
DEFINE_FWK_MODULE(SiStripApvGainDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
typedef DummyCondDBWriter<SiStripLorentzAngle,SiStripLorentzAngle,SiStripLorentzAngleRcd> SiStripLorentzAngleDummyDBWriter;
DEFINE_FWK_MODULE(SiStripLorentzAngleDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
typedef DummyCondDBWriter<SiStripBackPlaneCorrection,SiStripBackPlaneCorrection,SiStripBackPlaneCorrectionRcd> SiStripBackPlaneCorrectionDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBackPlaneCorrectionDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
typedef DummyCondDBWriter<SiStripThreshold,SiStripThreshold,SiStripThresholdRcd> SiStripThresholdDummyDBWriter;
DEFINE_FWK_MODULE(SiStripThresholdDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBadStripDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBadModuleDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBadFiberDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBadChannelDummyDBWriter);

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
typedef DummyCondDBWriter<SiStripQuality,SiStripBadStrip,SiStripQualityRcd> SiStripBadStripFromQualityDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBadStripFromQualityDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
typedef DummyCondDBWriter<SiStripDetVOff,SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffDummyDBWriter;
DEFINE_FWK_MODULE(SiStripDetVOffDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
typedef DummyCondDBWriter<SiStripLatency,SiStripLatency,SiStripLatencyRcd> SiStripLatencyDummyDBWriter;
DEFINE_FWK_MODULE(SiStripLatencyDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
typedef DummyCondDBWriter<SiStripBaseDelay,SiStripBaseDelay,SiStripBaseDelayRcd> SiStripBaseDelayDummyDBWriter;
DEFINE_FWK_MODULE(SiStripBaseDelayDummyDBWriter);

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
typedef DummyCondDBWriter<SiStripConfObject,SiStripConfObject, SiStripConfObjectRcd> SiStripConfObjectDummyDBWriter;
DEFINE_FWK_MODULE(SiStripConfObjectDummyDBWriter);

//---------------------------------------------------------------------------------------------------------------
// Dummy printers

#include "CalibTracker/SiStripESProducers/plugins/DBWriter/DummyCondObjPrinter.h"

typedef DummyCondObjPrinter<SiStripNoises,SiStripNoisesRcd> SiStripNoises_DecModeDummyPrinter;
DEFINE_FWK_MODULE(SiStripNoises_DecModeDummyPrinter);

typedef DummyCondObjPrinter<SiStripNoises,SiStripNoisesRcd> SiStripNoises_PeakModeDummyPrinter;
DEFINE_FWK_MODULE(SiStripNoises_PeakModeDummyPrinter);

typedef DummyCondObjPrinter<SiStripThreshold,SiStripThresholdRcd> SiStripThresholdDummyPrinter;
DEFINE_FWK_MODULE(SiStripThresholdDummyPrinter);

typedef DummyCondObjPrinter<SiStripThreshold,SiStripThresholdRcd> SiStripClusterThresholdDummyPrinter;
DEFINE_FWK_MODULE(SiStripClusterThresholdDummyPrinter);

typedef DummyCondObjPrinter<SiStripLorentzAngle,SiStripLorentzAngleRcd> SiStripLorentzAngleDummyPrinter;
DEFINE_FWK_MODULE(SiStripLorentzAngleDummyPrinter);

typedef DummyCondObjPrinter<SiStripBackPlaneCorrection,SiStripBackPlaneCorrectionRcd> SiStripBackPlaneCorrectionDummyPrinter;
DEFINE_FWK_MODULE(SiStripBackPlaneCorrectionDummyPrinter);

typedef DummyCondObjPrinter<SiStripPedestals,SiStripPedestalsRcd> SiStripPedestalsDummyPrinter;
DEFINE_FWK_MODULE(SiStripPedestalsDummyPrinter);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef DummyCondObjPrinter<SiStripApvGain,SiStripApvGainRcd> SiStripApvGainDummyPrinter;
DEFINE_FWK_MODULE(SiStripApvGainDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
typedef DummyCondObjPrinter<SiStripGain,SiStripGainRcd> SiStripGainDummyPrinter;
DEFINE_FWK_MODULE(SiStripGainDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
typedef DummyCondObjPrinter<SiStripGain,SiStripGainSimRcd> SiStripGainSimDummyPrinter;
DEFINE_FWK_MODULE(SiStripGainSimDummyPrinter);

typedef DummyCondObjPrinter<SiStripDetVOff,SiStripDetVOffRcd> SiStripDetVOffDummyPrinter;
DEFINE_FWK_MODULE(SiStripDetVOffDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
typedef DummyCondObjPrinter<SiStripDetCabling,SiStripDetCablingRcd> SiStripDetCablingDummyPrinter;
DEFINE_FWK_MODULE(SiStripDetCablingDummyPrinter);

typedef DummyCondObjPrinter<SiStripFedCabling,SiStripFedCablingRcd> SiStripFedCablingDummyPrinter;
DEFINE_FWK_MODULE(SiStripFedCablingDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadStripRcd> SiStripBadStripDummyPrinter;
DEFINE_FWK_MODULE(SiStripBadStripDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadModuleRcd> SiStripBadModuleDummyPrinter;
DEFINE_FWK_MODULE(SiStripBadModuleDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadFiberRcd> SiStripBadFiberDummyPrinter;
DEFINE_FWK_MODULE(SiStripBadFiberDummyPrinter);

typedef DummyCondObjPrinter<SiStripBadStrip,SiStripBadChannelRcd> SiStripBadChannelDummyPrinter;
DEFINE_FWK_MODULE(SiStripBadChannelDummyPrinter);

typedef DummyCondObjPrinter<SiStripLatency,SiStripLatencyRcd> SiStripLatencyDummyPrinter;
DEFINE_FWK_MODULE(SiStripLatencyDummyPrinter);

typedef DummyCondObjPrinter<SiStripBaseDelay,SiStripBaseDelayRcd> SiStripBaseDelayDummyPrinter;
DEFINE_FWK_MODULE(SiStripBaseDelayDummyPrinter);

#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"
typedef DummyCondObjPrinter<SiStripDelay,SiStripDelayRcd> SiStripDelayDummyPrinter;
DEFINE_FWK_MODULE(SiStripDelayDummyPrinter);

typedef DummyCondObjPrinter<SiStripLorentzAngle,SiStripLorentzAngleDepRcd> SiStripLorentzAngleDepDummyPrinter;
DEFINE_FWK_MODULE(SiStripLorentzAngleDepDummyPrinter);

typedef DummyCondObjPrinter<SiStripBackPlaneCorrection,SiStripBackPlaneCorrectionDepRcd> SiStripBackPlaneCorrectionDepDummyPrinter;
DEFINE_FWK_MODULE(SiStripBackPlaneCorrectionDepDummyPrinter);

typedef DummyCondObjPrinter<SiStripConfObject,SiStripConfObjectRcd> SiStripConfObjectDummyPrinter;
DEFINE_FWK_MODULE(SiStripConfObjectDummyPrinter);
