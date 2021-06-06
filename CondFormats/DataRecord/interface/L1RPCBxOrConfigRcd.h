#ifndef DataRecord_L1RPCBxOrConfigRcd_h
#define DataRecord_L1RPCBxOrConfigRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1RPCBxOrConfigRcd
//
/**\class L1RPCBxOrConfigRcd L1RPCBxOrConfigRcd.h CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/

#include "FWCore/Utilities/interface/mplVector.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1RPCBxOrConfigRcd
    : public edm::eventsetup::DependentRecordImplementation<L1RPCBxOrConfigRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
