#ifndef HcalFlagHFDigiTimeParamsRcd_H
#define HcalFlagHFDigiTimeParamsRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalFlagHFDigiTimeParamsRcd : public edm::eventsetup::DependentRecordImplementation<HcalFlagHFDigiTimeParamsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
