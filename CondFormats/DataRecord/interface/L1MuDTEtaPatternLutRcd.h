#ifndef L1MuDTEtaPatternLutRCD_H
#define L1MuDTEtaPatternLutRCD_H

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuDTEtaPatternLutRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuDTEtaPatternLutRcd> {};
class L1MuDTEtaPatternLutRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuDTEtaPatternLutRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
