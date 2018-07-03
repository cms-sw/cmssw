#ifndef Geometry_Records_PHGCalParametersRcd_H
#define Geometry_Records_PHGCalParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class PHGCalParametersRcd : public edm::eventsetup::DependentRecordImplementation<PHGCalParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // PHGCalParameters_H
