#ifndef DataRecord_L1TriggerKeyExtRcd_h
#define DataRecord_L1TriggerKeyExtRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

class L1TriggerKeyExtRcd
    : public edm::eventsetup::DependentRecordImplementation<L1TriggerKeyExtRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListExtRcd> > {};

#endif
