#ifndef HcalMCParamsRcd_H
#define HcalMCParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalMCParamsRcd : public edm::eventsetup::DependentRecordImplementation<
                            HcalMCParamsRcd,
                            edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord> > {};
#endif
