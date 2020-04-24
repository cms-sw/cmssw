#ifndef HcalFrontEndMapRcd_H
#define HcalFrontEndMapRcd_H
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalFrontEndMapRcd : public edm::eventsetup::DependentRecordImplementation<HcalFrontEndMapRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};
#endif
