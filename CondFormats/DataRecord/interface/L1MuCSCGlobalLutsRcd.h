#ifndef CondFormatsDataRecord_L1MuCSCGlobalLutsRcd_h
#define CondFormatsDataRecord_L1MuCSCGlobalLutsRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuCSCGlobalLutsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCGlobalLutsRcd> {};
class L1MuCSCGlobalLutsRcd : public edm::eventsetup::DependentRecordImplementation<L1MuCSCGlobalLutsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
