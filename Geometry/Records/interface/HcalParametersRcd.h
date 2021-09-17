#ifndef Geometry_Records_HcalParametersRcd_H
#define Geometry_Records_HcalParametersRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class HcalParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<HcalParametersRcd, edm::mpl::Vector<IdealGeometryRecord> > {
};

#endif  // Geometry_Records_HcalParameters_H
