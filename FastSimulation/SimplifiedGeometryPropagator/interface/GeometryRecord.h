#ifndef FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h
#define FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class GeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<GeometryRecord,
                                                            boost::mpl::vector<TrackerRecoGeometryRecord> > {};

#endif