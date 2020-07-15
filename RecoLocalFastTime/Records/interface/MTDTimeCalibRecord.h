#ifndef RecoLocalFastTime_Records_MTDTimeCalibRecord_h
#define RecoLocalFastTime_Records_MTDTimeCalibRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include <boost/mp11/list.hpp>

class MTDTimeCalibRecord
    : public edm::eventsetup::DependentRecordImplementation<MTDTimeCalibRecord,
                                                            boost::mp11::mp_list<MTDDigiGeometryRecord, MTDTopologyRcd> > {
};

#endif
