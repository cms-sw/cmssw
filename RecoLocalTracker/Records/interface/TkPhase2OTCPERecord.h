#ifndef RecoLocalTracker_Records_TkPhase2OTCPERecord_h
#define RecoLocalTracker_Records_TkPhase2OTCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"

#include "boost/mpl/vector.hpp"

class TkPhase2OTCPERecord
    : public edm::eventsetup::DependentRecordImplementation<
          TkPhase2OTCPERecord,
          boost::mpl::vector<TrackerDigiGeometryRecord, IdealMagneticFieldRecord, SiPhase2OuterTrackerLorentzAngleRcd> > {
};

#endif
