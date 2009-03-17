#ifndef CondFormats_SiStripCondDataRecords_h
#define CondFormats_SiStripCondDataRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class SiStripApvGainRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGainRcd> {};
class SiStripApvGainSimRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGainSimRcd> {};


class SiStripBadChannelRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadChannelRcd> {};
class SiStripBadFiberRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadFiberRcd> {};
class SiStripBadModuleRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadModuleRcd> {};
class SiStripBadStripRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadStripRcd> {};

class SiStripFedCablingRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripFedCablingRcd> {};

class SiStripLorentzAngleRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripLorentzAngleRcd> {};
class SiStripLorentzAngleSimRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripLorentzAngleSimRcd> {};

class SiStripModuleHVRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripModuleHVRcd> {};
class SiStripDCSStatusRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripDCSStatusRcd> {};

class SiStripNoisesRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripNoisesRcd> {};
class SiStripPedestalsRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripPedestalsRcd> {};

class SiStripPerformanceSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripPerformanceSummaryRcd> {};

class SiStripRunSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripRunSummaryRcd> {};

class SiStripSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripSummaryRcd> {};

class SiStripThresholdRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripThresholdRcd> {};

#endif
