// name: ratnikov, date: Mon Sep 26 17:02:44 CDT 2005
#ifndef HcalGainWidthsRcd_H
#define HcalGainWidthsRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalGainWidthsRcd : public edm::eventsetup::DependentRecordImplementation<HcalGainWidthsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
