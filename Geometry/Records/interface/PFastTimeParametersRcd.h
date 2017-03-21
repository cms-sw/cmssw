#ifndef Geometry_Records_PFastTimeParametersRcd_H
#define Geometry_Records_PFastTimeParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class PFastTimeParametersRcd : public edm::eventsetup::DependentRecordImplementation<PFastTimeParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // PFastTimeParameters_H
