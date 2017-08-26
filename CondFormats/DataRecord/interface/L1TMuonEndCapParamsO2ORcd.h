#ifndef L1TEMTFParamsRcd_L1TEMTFParamsO2ORcd_h
#define L1TEMTFParamsRcd_L1TEMTFParamsO2ORcd_h
// -*- C++ -*-
//
// Class  :     L1TMuonEndCapParamsRcd
//
// Author:      Matthew Carver
// Created:
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"

class L1TMuonEndCapParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonEndCapParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TMuonEndCapParamsRcd> > {};

#endif
