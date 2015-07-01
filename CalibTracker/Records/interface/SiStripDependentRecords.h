#ifndef CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H
#define CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

class SiStripFecCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripFecCablingRcd,
  boost::mpl::vector<SiStripFedCablingRcd> > {};

class SiStripDetCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripDetCablingRcd,
  boost::mpl::vector<SiStripFedCablingRcd,TrackerTopologyRcd,IdealGeometryRecord> > {};

class SiStripRegionCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripRegionCablingRcd,
  boost::mpl::vector<SiStripDetCablingRcd,TrackerDigiGeometryRecord,TrackerTopologyRcd> > {};

// class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainRcd, boost::mpl::vector<SiStripApvGainRcd> > {};
class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainRcd, boost::mpl::vector<SiStripApvGainRcd, SiStripApvGain2Rcd, SiStripApvGain3Rcd> > {};
class SiStripGainSimRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainSimRcd, boost::mpl::vector<SiStripApvGainSimRcd> > {};

class SiStripQualityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripQualityRcd, boost::mpl::vector<SiStripBadModuleRcd, SiStripBadFiberRcd, SiStripBadChannelRcd, SiStripBadStripRcd, SiStripDetCablingRcd, SiStripDCSStatusRcd, SiStripDetVOffRcd, RunInfoRcd> > {};

class SiStripDelayRcd : public edm::eventsetup::DependentRecordImplementation<SiStripDelayRcd, boost::mpl::vector<SiStripBaseDelayRcd> > {};

class SiStripLorentzAngleDepRcd : public edm::eventsetup::DependentRecordImplementation<SiStripLorentzAngleDepRcd, boost::mpl::vector<SiStripLatencyRcd, SiStripLorentzAngleRcd,IdealGeometryRecord> > {};

class SiStripBackPlaneCorrectionDepRcd : public edm::eventsetup::DependentRecordImplementation<SiStripBackPlaneCorrectionDepRcd, boost::mpl::vector<SiStripLatencyRcd, SiStripBackPlaneCorrectionRcd,IdealGeometryRecord> > {};

class SiStripHashedDetIdRcd : public edm::eventsetup::DependentRecordImplementation<SiStripHashedDetIdRcd, boost::mpl::vector<TrackerDigiGeometryRecord> > {};

class SiStripNoisesDepRcd : public edm::eventsetup::DependentRecordImplementation<SiStripNoisesDepRcd, boost::mpl::vector<SiStripNoisesRcd,IdealGeometryRecord> > {};

class SiStripBadModuleDepRcd : public edm::eventsetup::DependentRecordImplementation<SiStripBadModuleDepRcd, boost::mpl::vector<SiStripBadModuleRcd,IdealGeometryRecord> > {};

#endif 

