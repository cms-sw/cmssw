#ifndef CondFormatsDataRecord_L1GctChannelMaskRcd_h
#define CondFormatsDataRecord_L1GctChannelMaskRcd_h

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1GctChannelMaskRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GctChannelMaskRcd> {};
class L1GctChannelMaskRcd : public edm::eventsetup::DependentRecordImplementation<L1GctChannelMaskRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
