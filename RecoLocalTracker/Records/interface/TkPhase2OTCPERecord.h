#ifndef RecoLocalTracker_Records_TkPhase2OTCPERecord_h
#define RecoLocalTracker_Records_TkPhase2OTCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"
#include "FWCore/Utilities/interface/mplVector.h"

class TkPhase2OTCPERecord
    : public edm::eventsetup::DependentRecordImplementation<
          TkPhase2OTCPERecord,
          edm::mpl::Vector<TrackerDigiGeometryRecord, IdealMagneticFieldRecord, SiPhase2OuterTrackerLorentzAngleRcd> > {
};

#endif
