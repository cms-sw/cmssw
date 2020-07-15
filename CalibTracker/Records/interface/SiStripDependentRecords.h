#ifndef CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H
#define CALIBTRACKER_RECORDS_SISTRIPDEPENDENTRECORDS_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mp11/list.hpp>

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class SiStripFecCablingRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripFecCablingRcd,
                                                            boost::mp11::mp_list<SiStripFedCablingRcd> > {};

class SiStripDetCablingRcd : public edm::eventsetup::DependentRecordImplementation<
                                 SiStripDetCablingRcd,
                                 boost::mp11::mp_list<SiStripFedCablingRcd, TrackerTopologyRcd, IdealGeometryRecord> > {
};

class SiStripRegionCablingRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiStripRegionCablingRcd,
          boost::mp11::mp_list<SiStripDetCablingRcd, TrackerDigiGeometryRecord, TrackerTopologyRcd> > {};

// class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainRcd, boost::mp11::mp_list<SiStripApvGainRcd> > {};
class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<
                           SiStripGainRcd,
                           boost::mp11::mp_list<SiStripApvGainRcd, SiStripApvGain2Rcd, SiStripApvGain3Rcd> > {};
class SiStripGainSimRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripGainSimRcd,
                                                            boost::mp11::mp_list<SiStripApvGainSimRcd> > {};

class SiStripDelayRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripDelayRcd, boost::mp11::mp_list<SiStripBaseDelayRcd> > {
};

class SiStripLorentzAngleDepRcd : public edm::eventsetup::DependentRecordImplementation<
                                      SiStripLorentzAngleDepRcd,
                                      boost::mp11::mp_list<SiStripLatencyRcd, SiStripLorentzAngleRcd> > {};

class SiStripBackPlaneCorrectionDepRcd : public edm::eventsetup::DependentRecordImplementation<
                                             SiStripBackPlaneCorrectionDepRcd,
                                             boost::mp11::mp_list<SiStripLatencyRcd, SiStripBackPlaneCorrectionRcd> > {
};

class SiStripHashedDetIdRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripHashedDetIdRcd,
                                                            boost::mp11::mp_list<TrackerDigiGeometryRecord> > {};

class SiStripBadModuleFedErrRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripBadModuleFedErrRcd,
                                                            boost::mp11::mp_list<SiStripFedCablingRcd> > {};

class SiStripQualityRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripQualityRcd,
                                                            boost::mp11::mp_list<SiStripBadModuleRcd,
                                                                                 SiStripBadFiberRcd,
                                                                                 SiStripBadChannelRcd,
                                                                                 SiStripBadStripRcd,
                                                                                 SiStripDetCablingRcd,
                                                                                 SiStripDCSStatusRcd,
                                                                                 SiStripDetVOffRcd,
                                                                                 RunInfoRcd,
                                                                                 SiStripBadModuleFedErrRcd> > {};

#endif
