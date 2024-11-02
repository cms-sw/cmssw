#ifndef RecoLocalTracker_Records_FrameSoARecord_h
#define RecoLocalTracker_Records_FrameSoARecord_h
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class FrameSoARecord :public edm::eventsetup::DependentRecordImplementation<FrameSoARecord, edm::mpl::Vector<TrackerDigiGeometryRecord, TrackerTopologyRcd> > {};
#endif
