#ifndef DataRecord_L1RPCHsbConfigRcd_h
#define DataRecord_L1RPCHsbConfigRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1RPCHsbConfigRcd
// 
/**\class L1RPCHsbConfigRcd L1RPCHsbConfigRcd.h CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/

#include "boost/mpl/vector.hpp"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

class L1RPCHsbConfigRcd : public edm::eventsetup::DependentRecordImplementation<L1RPCHsbConfigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif

