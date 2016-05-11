#ifndef L1TGMTParamsRcd_L1TGMTParamsO2ORcd_h
#define L1TGMTParamsRcd_L1TGMTParamsO2ORcd_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     L1TMuonGlobalParamsRcd
// 
/**\class L1TMuonGlobalParamsRcd L1TMuonGlobalParamsRcd.h Subsystem/Package/interface/L1TMuonGlobalParamsRcd.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Author:      Thomas Reis
// Created:     Tue, 22 Sep 2015 13:32:54 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"

class L1TMuonGlobalParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonGlobalParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TMuonGlobalParamsRcd> > {};

#endif
