#ifndef RecoMuon_Record_MuonPerformanceRecord_h
#define RecoMuon_Record_MuonPerformanceRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include <boost/mp11/list.hpp>

class MuonPerformanceRecord : public edm::eventsetup::DependentRecordImplementation<
                                  MuonPerformanceRecord,
                                  boost::mp11::mp_list<PerformancePayloadRecord, PerformanceWPRecord> > {};

#endif
