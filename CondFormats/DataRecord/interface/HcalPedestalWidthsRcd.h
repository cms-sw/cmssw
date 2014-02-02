// name: ratnikov, date: Mon Sep 26 17:02:35 CDT 2005
#ifndef HcalPedestalWidthsRcd_H
#define HcalPedestalWidthsRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalPedestalWidthsRcd : public edm::eventsetup::DependentRecordImplementation<HcalPedestalWidthsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
