#ifndef RecoBTag_Record_BTagPerformanceRecord_h
#define RecoBTag_Record_BTagPerformanceRecord_h

#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "boost/mpl/vector.hpp"

class BTagPerformanceRecord
    : public edm::eventsetup::DependentRecordImplementation<
          BTagPerformanceRecord,
          boost::mpl::vector<PerformancePayloadRecord, PerformanceWPRecord>> {};

#endif
