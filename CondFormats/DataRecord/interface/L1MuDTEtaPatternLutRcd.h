#ifndef L1MuDTEtaPatternLutRCD_H
#define L1MuDTEtaPatternLutRCD_H

#include <boost/mp11/list.hpp>

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTEtaPatternLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTEtaPatternLutRcd> {};
class L1MuDTEtaPatternLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTEtaPatternLutRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
