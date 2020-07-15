#ifndef CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h
#define CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h

#include <boost/mp11/list.hpp>

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuGMTChannelMaskRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuGMTChannelMaskRcd,
                                                            boost::mp11::mp_list<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {
};

#endif
