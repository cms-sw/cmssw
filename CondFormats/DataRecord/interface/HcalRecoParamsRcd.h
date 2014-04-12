#ifndef HcalRecoParamsRcd_H
#define HcalRecoParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalRecoParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalRecoParamsRcd, boost::mpl::vector<IdealGeometryRecord> > {};
#endif
