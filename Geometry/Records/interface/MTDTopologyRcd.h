#ifndef Geometry_Records_MTDTopologyRcd
#define Geometry_Records_MTDTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "boost/mpl/vector.hpp"

class MTDTopologyRcd :
public edm::eventsetup::DependentRecordImplementation<MTDTopologyRcd,
  boost::mpl::vector<IdealGeometryRecord, PMTDParametersRcd> > {};

#endif
