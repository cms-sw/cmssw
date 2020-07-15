#ifndef FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h
#define FastSimulation_SimplifiedGeometryPropagator_GeometryRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include <boost/mp11/list.hpp>

class GeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<GeometryRecord,
                                                            boost::mp11::mp_list<TrackerRecoGeometryRecord> > {};

#endif