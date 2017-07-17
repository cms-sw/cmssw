#ifndef L1TEMTFParamsRcd_L1TEMTFParamsRcd_h
#define L1TEMTFParamsRcd_L1TEMTFParamsRcd_h
// -*- C++ -*-
//
// Class  :     L1TMuonEndcapParamsRcd
//
// Author:      Matthew Carver
// Created:
//

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

//class L1TMuonEndcapParamsRcd : public edm::eventsetup::DependentRecordImplementation<L1TMuonEndcapParamsRcd, boost::mpl::vector<L1TriggerKeyListExtRcd,L1TriggerKeyExtRcd> > {};

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TMuonEndcapParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TMuonEndcapParamsRcd> {};

#endif
