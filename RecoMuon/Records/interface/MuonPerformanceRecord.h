#ifndef RecoMuon_Record_MuonPerformanceRecord_h
#define RecoMuon_Record_MuonPerformanceRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MuonPerformanceRecord : public edm::eventsetup::DependentRecordImplementation<
                                  MuonPerformanceRecord,
                                  edm::mpl::Vector<PerformancePayloadRecord, PerformanceWPRecord> > {};

#endif
