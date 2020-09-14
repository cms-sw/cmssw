#ifndef CondFormatsDataRecord_L1MuGMTScalesRcd_h
#define CondFormatsDataRecord_L1MuGMTScalesRcd_h

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1MuGMTScalesRcd : public edm::eventsetup::EventSetupRecordImplementation<L1MuGMTScalesRcd> {};
class L1MuGMTScalesRcd
    : public edm::eventsetup::DependentRecordImplementation<L1MuGMTScalesRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
