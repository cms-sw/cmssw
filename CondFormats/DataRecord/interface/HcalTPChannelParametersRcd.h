#ifndef CondFormatsDataRecordHcalTPChannelParametersRcd_H
#define CondFormatsDataRecordHcalTPChannelParametersRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalTPChannelParametersRcd : public edm::eventsetup::DependentRecordImplementation<HcalTPChannelParametersRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};
#endif
