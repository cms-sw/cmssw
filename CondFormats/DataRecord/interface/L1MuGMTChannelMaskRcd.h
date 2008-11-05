#ifndef CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h
#define CondFormatsDataRecord_L1MuGMTChannelMaskRcd_h

#include "boost/mpl/vector.hpp"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuGMTChannelMaskRcd : public edm::eventsetup::DependentRecordImplementation<L1MuGMTChannelMaskRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
