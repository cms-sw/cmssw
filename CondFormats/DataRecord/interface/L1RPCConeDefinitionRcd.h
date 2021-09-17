#ifndef CondFormats_L1RPCConeDefinitionRcd_h
#define CondFormats_L1RPCConeDefinitionRcd_h
// -*- C++ -*-
//
// Package:     CondFormats
// Class  :     L1RPCConeDefinitionRcd
//

// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
// class L1RPCConeBuilderRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RPCConeBuilderRcd> {};

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include <FWCore/Utilities/interface/mplVector.h>
class L1RPCConeDefinitionRcd
    : public edm::eventsetup::DependentRecordImplementation<L1RPCConeDefinitionRcd,
                                                            edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd> > {};

#endif
