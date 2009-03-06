#ifndef RecoBTag_Record_BTagPerformanceRecord_h
#define RecoBTag_Record_BTagPerformanceRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/BTagPerformancePayloadRecord.h"        
#include "CondFormats/DataRecord/interface/BTagPerformanceWPRecord.h"        
#include "boost/mpl/vector.hpp"


class  BTagPerformanceRecord: public edm::eventsetup::DependentRecordImplementation<BTagPerformanceRecord,
			      boost::mpl::vector<BTagPerformancePayloadRecord,BTagPerformanceWPRecord> > {};

#endif 

