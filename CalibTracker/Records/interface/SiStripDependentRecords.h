#ifndef CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H
#define CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class SiStripFecCablingRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripFecCablingRcd,
                                                            edm::mpl::Vector<SiStripFedCablingRcd> > {};

class SiStripDetCablingRcd : public edm::eventsetup::DependentRecordImplementation<
                                 SiStripDetCablingRcd,
                                 edm::mpl::Vector<SiStripFedCablingRcd, TrackerTopologyRcd, IdealGeometryRecord> > {};

class SiStripRegionCablingRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiStripRegionCablingRcd,
          edm::mpl::Vector<SiStripDetCablingRcd, TrackerDigiGeometryRecord, TrackerTopologyRcd> > {};

// class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainRcd, edm::mpl::Vector<SiStripApvGainRcd> > {};
class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<
                           SiStripGainRcd,
                           edm::mpl::Vector<SiStripApvGainRcd, SiStripApvGain2Rcd, SiStripApvGain3Rcd> > {};
class SiStripGainSimRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripGainSimRcd, edm::mpl::Vector<SiStripApvGainSimRcd> > {
};

class SiStripDelayRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripDelayRcd, edm::mpl::Vector<SiStripBaseDelayRcd> > {};

class SiStripLorentzAngleDepRcd : public edm::eventsetup::DependentRecordImplementation<
                                      SiStripLorentzAngleDepRcd,
                                      edm::mpl::Vector<SiStripLatencyRcd, SiStripLorentzAngleRcd> > {};

class SiStripBackPlaneCorrectionDepRcd : public edm::eventsetup::DependentRecordImplementation<
                                             SiStripBackPlaneCorrectionDepRcd,
                                             edm::mpl::Vector<SiStripLatencyRcd, SiStripBackPlaneCorrectionRcd> > {};

class SiStripHashedDetIdRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripHashedDetIdRcd,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord> > {};

class SiStripQualityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripQualityRcd,
                                                                                edm::mpl::Vector<SiStripBadModuleRcd,
                                                                                                 SiStripBadFiberRcd,
                                                                                                 SiStripBadChannelRcd,
                                                                                                 SiStripBadStripRcd,
                                                                                                 SiStripDetCablingRcd,
                                                                                                 SiStripDCSStatusRcd,
                                                                                                 SiStripDetVOffRcd,
                                                                                                 RunInfoRcd> > {};

#endif
