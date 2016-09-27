#ifndef L1TBMTFParamsRcd_L1TBMTFParamsO2ORcd_h
#define L1TBMTFParamsRcd_L1TBMTFParamsO2ORcd_h
// -*- C++ -*-
//
// Class  :     L1TMuonBarrelParamsRcd
//
// Author:      Giannis Flouris
// Created:
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
class L1TMuonBarrelParamsO2ORcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonBarrelParamsO2ORcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd,L1TMuonBarrelParamsRcd> > {};

#endif
