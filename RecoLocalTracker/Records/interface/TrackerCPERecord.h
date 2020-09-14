#ifndef RecoLocalTracker_Records_TrackerCPERecord_h
#define RecoLocalTracker_Records_TrackerCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TrackerCPERecord : public edm::eventsetup::DependentRecordImplementation<
                             TrackerCPERecord,
                             edm::mpl::Vector<TrackerDigiGeometryRecord, IdealMagneticFieldRecord> > {};

#endif
