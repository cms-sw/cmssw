#ifndef CondFormatsDataRecord_L1MuTriggerPtScaleRcd_h
#define CondFormatsDataRecord_L1MuTriggerPtScaleRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuTriggerPtScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuTriggerPtScaleRcd> {};
class L1MuTriggerPtScaleRcd : public edm::eventsetup::DependentRecordImplementation<L1MuTriggerPtScaleRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
