#ifndef Geometry_Records_MTDTopologyRcd
#define Geometry_Records_MTDTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MTDTopologyRcd
    : public edm::eventsetup::DependentRecordImplementation<MTDTopologyRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord, PMTDParametersRcd> > {
};

#endif
