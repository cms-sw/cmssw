#ifndef Geometry_Records_PHcalParametersRcd_H
#define Geometry_Records_PHcalParametersRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class PHcalParametersRcd : public edm::eventsetup::DependentRecordImplementation<PHcalParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // Geometry_Records_PHcalParameters_H
