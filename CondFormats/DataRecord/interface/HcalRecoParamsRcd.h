#ifndef HcalRecoParamsRcd_H
#define HcalRecoParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalRecoParamsRcd : public edm::eventsetup::DependentRecordImplementation<
                              HcalRecoParamsRcd,
                              edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord> > {};
#endif
