#ifndef PTrackerAdditionalParametersPerDetRcd_H
#define PTrackerAdditionalParametersPerDetRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class PTrackerAdditionalParametersPerDetRcd
    : public edm::eventsetup::DependentRecordImplementation<PTrackerAdditionalParametersPerDetRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord>> {};
#endif
