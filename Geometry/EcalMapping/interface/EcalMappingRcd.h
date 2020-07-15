#ifndef ECALMAPPINGRCD_H
#define ECALMAPPINGRCD_H

#include <boost/mp11/list.hpp>

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

// class EcalMappingElectronicsRcd;

// class EcalMappingRcd : public edm::eventsetup::EventSetupRecordImplementation<EcalMappingRcd> {};

class EcalMappingRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalMappingRcd,
                                                            boost::mp11::mp_list<EcalMappingElectronicsRcd> > {};

#endif
