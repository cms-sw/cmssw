#ifndef L1TEMTFParamsRcd_L1TEMTFParamsO2ORcd_h
#define L1TEMTFParamsRcd_L1TEMTFParamsO2ORcd_h
// -*- C++ -*-
//
// Class  :     L1TMuonEndcapParamsRcd
//
// Author:      Matthew Carver
// Created:
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"

class L1TMuonEndcapParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonEndcapParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TMuonEndcapParamsRcd> > {};

#endif
