#ifndef L1TMTFOverlapFwVersionRcd_L1TMuonOverlapFwVersionO2ORcd_h
#define L1TMTFOverlapFwVersionRcd_L1TMuonOverlapFwVersionO2ORcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     L1TMuonOverlapFwVersionRcd
//
/**\class L1TMuonOverlapFwVersionRcd L1TMuonOverlapFwVersionRcd.h CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Author:      Michal Szleper
// Created:     Wed, 20 Oct 2020 13:55:50 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"
class L1TMuonOverlapFwVersionO2ORcd
    : public edm::eventsetup::DependentRecordImplementation<
          L1TMuonOverlapFwVersionO2ORcd,
          edm::mpl::Vector<L1TriggerKeyListExtRcd, L1TriggerKeyExtRcd, L1TMuonOverlapFwVersionRcd> > {};

#endif
