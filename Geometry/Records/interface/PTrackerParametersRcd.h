#ifndef PTrackerParametersRcd_H
#define PTrackerParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <boost/mp11/list.hpp>

class PTrackerParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PTrackerParametersRcd,
                                                            boost::mp11::mp_list<IdealGeometryRecord> > {};

#endif  // PTrackerParameters_H
