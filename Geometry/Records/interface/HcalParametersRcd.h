#ifndef Geometry_Records_HcalParametersRcd_H
#define Geometry_Records_HcalParametersRcd_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <boost/mp11/list.hpp>

class HcalParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<HcalParametersRcd,
                                                            boost::mp11::mp_list<IdealGeometryRecord> > {};

#endif  // Geometry_Records_HcalParameters_H
