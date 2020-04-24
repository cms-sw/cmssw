#ifndef HcalTimingParamsRcd_H
#define HcalTimingParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalTimingParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalTimingParamsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};
#endif
