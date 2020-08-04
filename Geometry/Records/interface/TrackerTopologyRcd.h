#ifndef Geometry_Records_TrackerTopologyRcd
#define Geometry_Records_TrackerTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class TrackerTopologyRcd : public edm::eventsetup::DependentRecordImplementation<
                               TrackerTopologyRcd,
                               edm::mpl::Vector<IdealGeometryRecord, PTrackerParametersRcd> > {};

#endif
