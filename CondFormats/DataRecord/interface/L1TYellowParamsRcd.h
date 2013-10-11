#ifndef CondFormatsDataRecord_L1TYellowParamsRcd_h
#define CondFormatsDataRecord_L1TYellowParamsRcd_h

// TODO Add Key list management:

//#include "boost/mpl/vector.hpp"

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1TYellowParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TYellowParamsRcd> {};
//class L1TYellowParamsRcd : public edm::eventsetup::DependentRecordImplementation<L1TYellowParamsRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
