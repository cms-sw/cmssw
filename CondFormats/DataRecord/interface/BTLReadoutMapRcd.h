#ifndef CondFormats_DataRecord_BTLReadoutMapRcd_h
#define CondFormats_DataRecord_BTLReadoutMapRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class BTLReadoutMapRcd
    : public edm::eventsetup::DependentRecordImplementation<BTLReadoutMapRcd,
                                                            edm::mpl::Vector<MTDDigiGeometryRecord, MTDTopologyRcd> > {
};
#endif
