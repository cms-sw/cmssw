#ifndef CondFormats_DataRecord_HcalPFCutsRcd_h
#define CondFormats_DataRecord_HcalPFCutsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalPFCutsRcd
    : public edm::eventsetup::
          DependentRecordImplementation<HcalPFCutsRcd, edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord> > {
};

#endif  // CondFormats_DataRecord_HcalPFCutsRcd_h
