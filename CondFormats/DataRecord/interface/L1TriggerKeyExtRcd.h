#ifndef DataRecord_L1TriggerKeyExtRcd_h
#define DataRecord_L1TriggerKeyExtRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

class L1TriggerKeyExtRcd
    : public edm::eventsetup::DependentRecordImplementation<L1TriggerKeyExtRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListExtRcd> > {};

#endif
