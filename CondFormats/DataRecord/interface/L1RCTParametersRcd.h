#ifndef CondFormatsDataRecord_L1RCTParametersRcd_h
#define CondFormatsDataRecord_L1RCTParametersRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RCTParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RCTParametersRcd> {};
class L1RCTParametersRcd : public edm::eventsetup::DependentRecordImplementation<L1RCTParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
