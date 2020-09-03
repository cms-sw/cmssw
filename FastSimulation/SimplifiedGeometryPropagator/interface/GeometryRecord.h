#ifndef FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h
#define FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class GeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<GeometryRecord,
                                                            edm::mpl::Vector<TrackerRecoGeometryRecord> > {};

#endif