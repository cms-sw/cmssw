#ifndef PTrackerParametersRcd_H
#define PTrackerParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class PTrackerParametersRcd : public edm::eventsetup::DependentRecordImplementation<PTrackerParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // PTrackerParameters_H
