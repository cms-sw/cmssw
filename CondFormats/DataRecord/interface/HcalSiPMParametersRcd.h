#ifndef CondFormatsDataRecordHcalSiPMParametersRcd_H
#define CondFormatsDataRecordHcalSiPMParametersRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalSiPMParametersRcd : public edm::eventsetup::DependentRecordImplementation<HcalSiPMParametersRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};
#endif
