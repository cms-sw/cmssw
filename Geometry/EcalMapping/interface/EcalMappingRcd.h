#ifndef ECALMAPPINGRCD_H
#define ECALMAPPINGRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

// class EcalMappingElectronicsRcd;

// class EcalMappingRcd : public edm::eventsetup::EventSetupRecordImplementation<EcalMappingRcd> {};

class EcalMappingRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalMappingRcd,
                                                            edm::mpl::Vector<EcalMappingElectronicsRcd> > {};

#endif
