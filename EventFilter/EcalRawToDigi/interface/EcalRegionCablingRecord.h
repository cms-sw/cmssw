#ifndef EcalRegionCablingRecord_H
#define EcalRegionCablingRecord_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include <boost/mp11/list.hpp>

class EcalRegionCablingRecord
    : public edm::eventsetup::DependentRecordImplementation<EcalRegionCablingRecord,
                                                            boost::mp11::mp_list<EcalMappingRcd> > {};

#endif
