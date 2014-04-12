#ifndef HcalMCParamsRcd_H
#define HcalMCParamsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalMCParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalMCParamsRcd, boost::mpl::vector<IdealGeometryRecord> > {};
#endif
