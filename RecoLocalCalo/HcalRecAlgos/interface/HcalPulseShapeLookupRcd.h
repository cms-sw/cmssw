#ifndef RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h
#define RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/HcalInterpolatedPulseMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalPulseDelaysRcd.h"

class HcalPulseShapeLookupRcd : public edm::eventsetup::DependentRecordImplementation<
                                    HcalPulseShapeLookupRcd,
                                    edm::mpl::Vector<HcalRecNumberingRecord,
						     CaloGeometryRecord,
						     HcalInterpolatedPulseMapRcd,
						     HcalPulseDelaysRcd> > {};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalPulseShapeLookupRcd_h
