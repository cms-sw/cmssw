#ifndef RecoLocalFastTime_Records_MTDTimeCalibRecord_h
#define RecoLocalFastTime_Records_MTDTimeCalibRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class MTDTimeCalibRecord
    : public edm::eventsetup::DependentRecordImplementation<MTDTimeCalibRecord,
                                                            edm::mpl::Vector<MTDDigiGeometryRecord, MTDTopologyRcd> > {
};

#endif
