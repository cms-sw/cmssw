#ifndef CondFormatsDataRecord_L1RCTNoisyChannelMaskRcd_h
#define CondFormatsDataRecord_L1RCTNoisyChannelMaskRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RCTChannelMaskRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RCTChannelMaskRcd> {};
class L1RCTNoisyChannelMaskRcd : public edm::eventsetup::DependentRecordImplementation<L1RCTNoisyChannelMaskRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
