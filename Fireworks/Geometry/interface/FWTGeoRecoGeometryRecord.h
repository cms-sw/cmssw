#ifndef GEOMETRY_FWTGEORECO_GEOMETRY_RECORD_H
# define GEOMETRY_FWTGEORECO_GEOMETRY_RECORD_H

# include "FWCore/Framework/interface/DependentRecordImplementation.h"
# include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
# include "Geometry/Records/interface/CaloGeometryRecord.h"
# include "Geometry/Records/interface/TrackerTopologyRcd.h"

class FWTGeoRecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<FWTGeoRecoGeometryRecord,boost::mpl::vector<GlobalTrackingGeometryRecord,CaloGeometryRecord,TrackerTopologyRcd> > {};

#endif // GEOMETRY_FWTGEORECO_GEOMETRY_RECORD_H
