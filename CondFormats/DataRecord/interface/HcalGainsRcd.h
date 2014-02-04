// name: ratnikov, date: Mon Sep 26 17:02:30 CDT 2005
#ifndef HcalGainsRcd_H
#define HcalGainsRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalGainsRcd : public edm::eventsetup::DependentRecordImplementation<HcalGainsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
