#ifndef Geometry_CaloEventSetyp_CaloTopologyRecord
#define Geometry_CaloEventSetyp_CaloTopologyRecord

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "boost/mpl/vector.hpp"


class CaloTopologyRecord : public edm::eventsetup::DependentRecordImplementation<CaloTopologyRecord,
  boost::mpl::vector<IdealGeometryRecord> > {};

#endif 
