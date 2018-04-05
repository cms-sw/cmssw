#ifndef CondFormats_SiStripCondDataRecords_h
#define CondFormats_SiStripCondDataRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

/*Recod associated to SiStripApvGain Object: the SimRcd is used in simulation only*/
class SiStripApvGainRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGainRcd> {};
class SiStripApvGain2Rcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGain2Rcd> {};
class SiStripApvGain3Rcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGain3Rcd> {};
class SiStripApvGainSimRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripApvGainSimRcd> {};

/*Record associated to SiStripBadStrip Object*/
class SiStripBadChannelRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadChannelRcd> {};
class SiStripBadFiberRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadFiberRcd> {};
class SiStripBadModuleRcd : public edm::eventsetup::DependentRecordImplementation<SiStripBadModuleRcd, boost::mpl::vector<TrackerTopologyRcd> > {};
class SiStripBadStripRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBadStripRcd> {};
class SiStripDCSStatusRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripDCSStatusRcd> {};

class SiStripFedCablingRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripFedCablingRcd> {};

/*Recod associated to SiStripLorenzaAngle Object: the SimRcd is used in simulation only*/
class SiStripLorentzAngleRcd : public edm::eventsetup::DependentRecordImplementation<SiStripLorentzAngleRcd, boost::mpl::vector<TrackerTopologyRcd> > {};
class SiStripLorentzAngleSimRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripLorentzAngleSimRcd> {};

class SiStripBackPlaneCorrectionRcd : public edm::eventsetup::DependentRecordImplementation<SiStripBackPlaneCorrectionRcd, boost::mpl::vector<TrackerTopologyRcd> > {};

class SiStripDetVOffRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripDetVOffRcd> {};

class SiStripLatencyRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripLatencyRcd> {};

class SiStripBaseDelayRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripBaseDelayRcd> {};

class SiStripNoisesRcd : public edm::eventsetup::DependentRecordImplementation<SiStripNoisesRcd, boost::mpl::vector<TrackerTopologyRcd> > {};

class SiStripPedestalsRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripPedestalsRcd> {};

class SiStripRunSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripRunSummaryRcd> {};

class SiStripSummaryRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripSummaryRcd> {};

/*Record Associated to SiStripThreshold Object*/
class SiStripThresholdRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripThresholdRcd> {};
class SiStripClusterThresholdRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripClusterThresholdRcd> {};

/*Record for the configuration object*/
class SiStripConfObjectRcd : public edm::eventsetup::EventSetupRecordImplementation<SiStripConfObjectRcd> {};

/*Records for upgrade */
class Phase2TrackerCablingRcd : public edm::eventsetup::EventSetupRecordImplementation<Phase2TrackerCablingRcd> {};

#endif
