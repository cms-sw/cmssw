#ifndef GEOMETRY_FWRECO_GEOMETRY_RECORD_H
# define GEOMETRY_FWRECO_GEOMETRY_RECORD_H

# include "FWCore/Framework/interface/DependentRecordImplementation.h"
# include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
# include "Geometry/Records/interface/CaloGeometryRecord.h"

class FWRecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<FWRecoGeometryRecord,boost::mpl::vector<MuonGeometryRecord, GlobalTrackingGeometryRecord,CaloGeometryRecord> > {};

#endif // GEOMETRY_FWRECO_GEOMETRY_RECORD_H
