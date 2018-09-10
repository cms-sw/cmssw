#ifndef PMTDParametersRcd_H
#define PMTDParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class PMTDParametersRcd : public edm::eventsetup::DependentRecordImplementation<PMTDParametersRcd,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif // PMTDParameters_H
