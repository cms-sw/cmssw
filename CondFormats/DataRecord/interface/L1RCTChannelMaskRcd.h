#ifndef CondFormatsDataRecord_L1RCTChannelMaskRcd_h
#define CondFormatsDataRecord_L1RCTChannelMaskRcd_h

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RCTChannelMaskRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RCTChannelMaskRcd> {};
class L1RCTChannelMaskRcd
    : public edm::eventsetup::DependentRecordImplementation<L1RCTChannelMaskRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
