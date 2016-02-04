#ifndef Geometry_Records_CaloTopologyRecord
#define Geometry_Records_CaloTopologyRecord

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "boost/mpl/vector.hpp"


class CaloTopologyRecord : 
   public edm::eventsetup::DependentRecordImplementation<
   CaloTopologyRecord,
                     boost::mpl::vector<
                     CaloGeometryRecord> > {};

#endif 
