#ifndef Geometry_Records_MTDTopologyRcd
#define Geometry_Records_MTDTopologyRcd

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include <boost/mp11/list.hpp>

class MTDTopologyRcd
    : public edm::eventsetup::DependentRecordImplementation<MTDTopologyRcd,
                                                            boost::mp11::mp_list<IdealGeometryRecord, PMTDParametersRcd> > {
};

#endif
