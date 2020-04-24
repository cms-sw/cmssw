#ifndef L1TMTFOverlapParamsRcd_L1TMTFOverlapParamsO2ORcd_h
#define L1TMTFOverlapParamsRcd_L1TMTFOverlapParamsO2ORcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     L1TMuonOverlapParamsRcd
// 
/**\class L1TMuonOverlapParamsRcd L1TMuonOverlapParamsRcd.h CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Author:      Artur Kalinowski
// Created:     Tue, 06 Oct 2015 11:46:55 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
class L1TMuonOverlapParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonOverlapParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TMuonOverlapParamsRcd> > {};

//class L1TMuonOverlapParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TMuonOverlapParamsRcd> {};

#endif
