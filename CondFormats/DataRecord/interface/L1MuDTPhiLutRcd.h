#ifndef L1MuDTPhiLutRCD_H
#define L1MuDTPhiLutRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTPhiLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTPhiLutRcd> {};
class L1MuDTPhiLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTPhiLutRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
