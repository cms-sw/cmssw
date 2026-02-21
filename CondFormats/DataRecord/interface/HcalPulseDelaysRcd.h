#ifndef CondFormats_DataRecord_HcalPulseDelaysRcd_h
#define CondFormats_DataRecord_HcalPulseDelaysRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalPulseDelaysRcd : public edm::eventsetup::DependentRecordImplementation<
                               HcalPulseDelaysRcd,
                               edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord> > {};

#endif  // CondFormats_DataRecord_HcalPulseDelaysRcd_h
