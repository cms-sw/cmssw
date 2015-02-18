#ifndef Geometry_Records_TrackerTopologyRcd
#define Geometry_Records_TrackerTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "boost/mpl/vector.hpp"

class TrackerTopologyRcd :
public edm::eventsetup::DependentRecordImplementation<TrackerTopologyRcd,
  boost::mpl::vector<PTrackerParametersRcd> > {};

#endif
