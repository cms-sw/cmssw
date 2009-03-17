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
struct CabRcdName{ static const char* name(){return "SiStripFedCablingRcd";}};
typedef DummyCondDBWriter<SiStripFedCabling,SiStripFedCabling,SiStripFedCablingRcd,CabRcdName> SiStripFedCablingDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripFedCablingDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
struct PedRcdName{ static const char* name(){return "SiStripPedestalsRcd";}};
typedef DummyCondDBWriter<SiStripPedestals,SiStripPedestals,SiStripPedestalsRcd,PedRcdName> SiStripPedestalsDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripPedestalsDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
struct NoiseRcdName{ static const char* name(){return "SiStripNoisesRcd";}};
typedef DummyCondDBWriter<SiStripNoises,SiStripNoises,SiStripNoisesRcd,NoiseRcdName> SiStripNoisesDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripNoisesDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
struct GainRcdName{ static const char* name(){return "SiStripApvGainRcd";}};
typedef DummyCondDBWriter<SiStripApvGain,SiStripApvGain,SiStripApvGainRcd,GainRcdName> SiStripApvGainDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripApvGainDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
struct LARcdName{ static const char* name(){return "SiStripLorentzAngleRcd";}};
typedef DummyCondDBWriter<SiStripLorentzAngle,SiStripLorentzAngle,SiStripLorentzAngleRcd,LARcdName> SiStripLorentzAngleDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripLorentzAngleDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
struct ThRcdName{ static const char* name(){return "SiStripThresholdRcd";}};
typedef DummyCondDBWriter<SiStripThreshold,SiStripThreshold,SiStripThresholdRcd,ThRcdName> SiStripThresholdDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdDummyDBWriter);


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
struct BadStripRcdName{ static const char* name(){return "SiStripBadStrip";}};
typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadStripRcd,BadStripRcdName> SiStripBadStripDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadModuleRcd,BadStripRcdName> SiStripBadModuleDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadModuleDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadFiberRcd,BadStripRcdName> SiStripBadFiberDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadFiberDummyDBWriter);

typedef DummyCondDBWriter<SiStripBadStrip,SiStripBadStrip,SiStripBadChannelRcd,BadStripRcdName> SiStripBadChannelDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadChannelDummyDBWriter);

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
typedef DummyCondDBWriter<SiStripQuality,SiStripBadStrip,SiStripQualityRcd,BadStripRcdName> SiStripBadStripFromQualityDummyDBWriter;
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripFromQualityDummyDBWriter);
