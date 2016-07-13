#ifndef DataRecord_L1TriggerKeyExtRcd_h
#define DataRecord_L1TriggerKeyExtRcd_h

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

class L1TriggerKeyExtRcd : public edm::eventsetup::DependentRecordImplementation<L1TriggerKeyExtRcd, boost::mpl::vector<L1TriggerKeyListExtRcd> > {};

#endif
