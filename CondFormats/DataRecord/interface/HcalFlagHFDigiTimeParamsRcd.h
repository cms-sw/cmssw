#ifndef HcalFlagHFDigiTimeParamsRcd_H
#define HcalFlagHFDigiTimeParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalFlagHFDigiTimeParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalFlagHFDigiTimeParamsRcd, boost::mpl::vector<IdealGeometryRecord> > {};
#endif
