#ifndef L1Scales_L1HtMissScaleRcd_h
#define L1Scales_L1HtMissScaleRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1HtMissScaleRcd
// 
/**\class L1HtMissScaleRcd L1HtMissScaleRcd.h CondFormats/DataRecord/interface/L1HtMissScaleRcd.h

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

//class L1HtMissScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1HtMissScaleRcd> {};
class L1HtMissScaleRcd : public edm::eventsetup::DependentRecordImplementation<L1HtMissScaleRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
