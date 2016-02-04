#ifndef CondFormatsDataRecord_L1MuCSCTFConfigurationRcd_h

#define CondFormatsDataRecord_L1MuCSCTFConfigurationRcd_h


#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"



//class L1MuCSCTFConfigurationRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCTFConfigurationRcd> {};
class L1MuCSCTFConfigurationRcd : public edm::eventsetup::DependentRecordImplementation<L1MuCSCTFConfigurationRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};



#endif

