#ifndef PMTDParametersRcd_H
#define PMTDParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <boost/mp11/list.hpp>

class PMTDParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PMTDParametersRcd,
                                                            boost::mp11::mp_list<IdealGeometryRecord> > {};

#endif  // PMTDParameters_H
