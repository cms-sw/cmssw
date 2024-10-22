#ifndef CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h
#define CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h

#include "FWCore/Utilities/interface/mplVector.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuGMTChannelMaskRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuGMTChannelMaskRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
