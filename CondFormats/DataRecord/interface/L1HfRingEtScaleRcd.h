#ifndef L1Scales_L1HfRingEtScaleRcd_h
#define L1Scales_L1HfRingEtScaleRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1HfRingEtScaleRcd
// 
/**\class L1HfRingEtScaleRcd L1HfRingEtScaleRcd.h CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Jim Brooke
// Created:     Wed Oct  4 16:49:43 CEST 2006
// $Id: 
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1HfRingEtScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1HfRingEtScaleRcd> {};
class L1HfRingEtScaleRcd : public edm::eventsetup::DependentRecordImplementation<L1HfRingEtScaleRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
