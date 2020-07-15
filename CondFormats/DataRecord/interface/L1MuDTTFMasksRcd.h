#ifndef L1MuDTTFMasksRCD_H
#define L1MuDTTFMasksRCD_H

#include <boost/mp11/list.hpp>

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuDTTFMasksRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTTFMasksRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
