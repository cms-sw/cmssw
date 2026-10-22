#ifndef CondFormats_SiPhase2InnerTrackerCondDataRecords_h
#define CondFormats_SiPhase2InnerTrackerCondDataRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

/*Record associated to SiPixelQuality (IT) Object:*/
class SiPhase2InnerTrackerBadModuleRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPhase2InnerTrackerBadModuleRcd,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord> > {};
#endif
