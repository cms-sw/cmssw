#ifndef Geometry_Records_TrackerTopologyRcd
#define Geometry_Records_TrackerTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include <boost/mp11/list.hpp>

class TrackerTopologyRcd : public edm::eventsetup::DependentRecordImplementation<
                               TrackerTopologyRcd,
                               boost::mp11::mp_list<IdealGeometryRecord, PTrackerParametersRcd> > {};

#endif
