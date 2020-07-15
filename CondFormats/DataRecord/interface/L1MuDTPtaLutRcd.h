#ifndef L1MuDTPtaLutRCD_H
#define L1MuDTPtaLutRCD_H

#include <boost/mp11/list.hpp>

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTPtaLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTPtaLutRcd> {};
class L1MuDTPtaLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTPtaLutRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
