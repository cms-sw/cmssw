#ifndef RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h
#define RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class HcalPulseShapeLookupRcd
    : public edm::eventsetup::
          DependentRecordImplementation<HcalPulseShapeLookupRcd, edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord> > {
};

#endif // RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h
