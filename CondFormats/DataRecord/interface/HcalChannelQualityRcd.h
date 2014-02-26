// name: ratnikov, date: Thu Oct 20 01:13:51 CDT 2005
#ifndef HcalChannelQualityRcd_H
#define HcalChannelQualityRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalChannelQualityRcd : public edm::eventsetup::DependentRecordImplementation<HcalChannelQualityRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
