#ifndef Geometry_Records_PFastTimeParametersRcd_H
#define Geometry_Records_PFastTimeParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class PFastTimeParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PFastTimeParametersRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord> > {};

#endif  // PFastTimeParameters_H
