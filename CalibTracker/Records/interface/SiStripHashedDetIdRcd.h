#ifndef CalibTracker_Records_SiStripHashedDetIdRcd_h
#define CalibTracker_Records_SiStripHashedDetIdRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "boost/mpl/vector.hpp"

/** 
    @class SiStripHashedDetIdRcd
    @author R.Bainbridge
*/
class SiStripHashedDetIdRcd : public edm::eventsetup::DependentRecordImplementation<SiStripHashedDetIdRcd, boost::mpl::vector<TrackerDigiGeometryRecord> > {};

#endif // CalibTracker_Records_SiStripHashedDetIdRcd_h

