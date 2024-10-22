#ifndef L1MuDTTFParametersRCD_H
#define L1MuDTTFParametersRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1MuDTTFParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTTFParametersRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
