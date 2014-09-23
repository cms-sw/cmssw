#ifndef ZDCLowGainFractionsRcd_H
#define ZDCLowGainFractionsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class ZDCLowGainFractionsRcd : public edm::eventsetup::DependentRecordImplementation<ZDCLowGainFractionsRcd, boost::mpl::vector<IdealGeometryRecord> > {};
#endif

