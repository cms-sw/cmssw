#ifndef HcalLongRecoParamsRcd_H
#define HcalLongRecoParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalLongRecoParamsRcd : public edm::eventsetup::DependentRecordImplementation<
                                  HcalLongRecoParamsRcd,
                                  boost::mp11::mp_list<HcalRecNumberingRecord, IdealGeometryRecord> > {};
#endif
