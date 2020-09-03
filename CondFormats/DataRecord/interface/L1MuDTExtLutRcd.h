#ifndef L1MuDTExtLutRCD_H
#define L1MuDTExtLutRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTExtLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTExtLutRcd> {};
class L1MuDTExtLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTExtLutRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
