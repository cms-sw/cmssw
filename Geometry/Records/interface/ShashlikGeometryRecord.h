#ifndef RECORDS_SHASHLIK_GEOMETRY_RECORD_H
# define RECORDS_SHASHLIK_GEOMETRY_RECORD_H

# include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
# include "FWCore/Framework/interface/DependentRecordImplementation.h"
# include "Geometry/Records/interface/IdealGeometryRecord.h"
# include "Geometry/Records/interface/ShashlikNumberingRecord.h"

class ShashlikGeometryRecord : 
  public edm::eventsetup::DependentRecordImplementation<
  ShashlikGeometryRecord,
  boost::mpl::vector<
    IdealGeometryRecord,
    ShashlikNumberingRecord > > {};

#endif // RECORDS_SHASHLIK_GEOMETRY_RECORD_H
