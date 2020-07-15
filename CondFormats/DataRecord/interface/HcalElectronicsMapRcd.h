#ifndef HcalElectronicsMapRcd_H
#define HcalElectronicsMapRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalElectronicsMapRcd : public edm::eventsetup::DependentRecordImplementation<
                                  HcalElectronicsMapRcd,
                                  boost::mp11::mp_list<HcalRecNumberingRecord, IdealGeometryRecord> > {};

#endif
