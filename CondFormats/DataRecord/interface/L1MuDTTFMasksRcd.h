#ifndef L1MuDTTFMasksRCD_H
#define L1MuDTTFMasksRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuDTTFMasksRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTTFMasksRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
