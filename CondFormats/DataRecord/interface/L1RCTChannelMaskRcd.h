#ifndef CondFormatsDataRecord_L1RCTChannelMaskRcd_h
#define CondFormatsDataRecord_L1RCTChannelMaskRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RCTChannelMaskRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RCTChannelMaskRcd> {};
class L1RCTChannelMaskRcd : public edm::eventsetup::DependentRecordImplementation<L1RCTChannelMaskRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
