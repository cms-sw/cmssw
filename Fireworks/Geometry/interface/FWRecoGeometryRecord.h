#ifndef GEOMETRY_FWRECO_GEOMETRY_RECORD_H
#define GEOMETRY_FWRECO_GEOMETRY_RECORD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/FastTimeGeometryRecord.h"

class FWRecoGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<
          FWRecoGeometryRecord,
          edm::mpl::Vector<MuonGeometryRecord, GlobalTrackingGeometryRecord, CaloGeometryRecord> > {};

#endif  // GEOMETRY_FWRECO_GEOMETRY_RECORD_H
