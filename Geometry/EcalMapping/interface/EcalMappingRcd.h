#ifndef ECALMAPPINGRCD_H
#define ECALMAPPINGRCD_H

#include "boost/mpl/vector.hpp"


#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

// class EcalMappingElectronicsRcd;


// class EcalMappingRcd : public edm::eventsetup::EventSetupRecordImplementation<EcalMappingRcd> {};

class EcalMappingRcd : public edm::eventsetup::DependentRecordImplementation <EcalMappingRcd, 
             boost::mpl::vector<EcalMappingElectronicsRcd> >  {};



#endif

