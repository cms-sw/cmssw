#ifndef PTrackerParametersRcd_H
#define PTrackerParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class PTrackerParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PTrackerParametersRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord> > {};

#endif  // PTrackerParameters_H
