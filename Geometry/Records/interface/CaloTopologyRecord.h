#ifndef Geometry_Records_CaloTopologyRecord
#define Geometry_Records_CaloTopologyRecord

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class CaloTopologyRecord
    : public edm::eventsetup::DependentRecordImplementation<CaloTopologyRecord, edm::mpl::Vector<CaloGeometryRecord> > {
};

#endif
