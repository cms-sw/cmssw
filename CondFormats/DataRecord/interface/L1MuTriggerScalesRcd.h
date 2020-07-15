#ifndef CondFormatsDataRecord_L1MuTriggerScalesRcd_h
#define CondFormatsDataRecord_L1MuTriggerScalesRcd_h

#include <boost/mp11/list.hpp>

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuTriggerScalesRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuTriggerScalesRcd> {};
class L1MuTriggerScalesRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuTriggerScalesRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
