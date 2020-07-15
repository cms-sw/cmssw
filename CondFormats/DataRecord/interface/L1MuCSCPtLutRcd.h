#ifndef CondFormatsDataRecord_L1MuCSCPtLutRcd_h
#define CondFormatsDataRecord_L1MuCSCPtLutRcd_h

#include <boost/mp11/list.hpp>

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuCSCPtLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuCSCPtLutRcd> {};
class L1MuCSCPtLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuCSCPtLutRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
