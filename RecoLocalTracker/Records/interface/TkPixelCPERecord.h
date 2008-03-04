#ifndef RecoLocalTracker_Records_TkPixelCPERecord_h
#define RecoLocalTracker_Records_TkPixelCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"                

#include "boost/mpl/vector.hpp"

class  TkPixelCPERecord: public edm::eventsetup::DependentRecordImplementation<TkPixelCPERecord,
  boost::mpl::vector<TrackerDigiGeometryRecord,IdealMagneticFieldRecord> > {};

#endif 

