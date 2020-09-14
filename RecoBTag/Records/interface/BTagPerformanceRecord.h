#ifndef RecoBTag_Record_BTagPerformanceRecord_h
#define RecoBTag_Record_BTagPerformanceRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class BTagPerformanceRecord : public edm::eventsetup::DependentRecordImplementation<
                                  BTagPerformanceRecord,
                                  edm::mpl::Vector<PerformancePayloadRecord, PerformanceWPRecord> > {};

#endif
