#ifndef CondFormatsDataRecord_L1RCTNoisyChannelMaskRcd_h
#define CondFormatsDataRecord_L1RCTNoisyChannelMaskRcd_h

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RCTChannelMaskRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RCTChannelMaskRcd> {};
class L1RCTNoisyChannelMaskRcd
    : public edm::eventsetup::DependentRecordImplementation<L1RCTNoisyChannelMaskRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
