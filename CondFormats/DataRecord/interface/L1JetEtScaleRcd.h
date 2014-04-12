#ifndef L1Scales_L1JetEtScaleRcd_h
#define L1Scales_L1JetEtScaleRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1JetEtScaleRcd
// 
/**\class L1JetEtScaleRcd L1JetEtScaleRcd.h CondFormats/DataRecord/interface/L1JetEtScaleRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Oct  4 16:49:43 CEST 2006
// $Id: L1JetEtScaleRcd.h,v 1.1 2007/03/16 13:51:29 heath Exp $
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1JetEtScaleRcd : public edm::eventsetup::EventSetupRecordImplementation<L1JetEtScaleRcd> {};
class L1JetEtScaleRcd : public edm::eventsetup::DependentRecordImplementation<L1JetEtScaleRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};


#endif
