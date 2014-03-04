#ifndef HcalRecoParamsRcd_H
#define HcalRecoParamsRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalRecoParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalRecoParamsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
