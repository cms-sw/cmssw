#ifndef RecoLocalFastTime_Records_MTDCPERecord_h
#define RecoLocalFastTime_Records_MTDCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include <boost/mp11/list.hpp>

class MTDCPERecord
    : public edm::eventsetup::DependentRecordImplementation<MTDCPERecord, boost::mp11::mp_list<MTDDigiGeometryRecord> > {
};

#endif
