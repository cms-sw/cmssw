#ifndef Geometry_Records_HcalParametersRcd_H
#define Geometry_Records_HcalParametersRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class HcalParametersRcd : public edm::eventsetup::DependentRecordImplementation<HcalParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // Geometry_Records_HcalParameters_H
