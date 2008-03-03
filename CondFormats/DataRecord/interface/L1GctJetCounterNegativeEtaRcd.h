#ifndef DataRecord_L1GctJetCounterNegativeEtaRcd_h
#define DataRecord_L1GctJetCounterNegativeEtaRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1GctJetCounterNegativeEtaRcd
// 
/**\class L1GctJetCounterNegativeEtaRcd L1GctJetCounterNegativeEtaRcd.h CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Tue Jul 10 10:14:03 CEST 2007
// $Id: L1GctJetCounterNegativeEtaRcd.h,v 1.1 2007/09/17 11:22:33 heath Exp $
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1GctJetCounterNegativeEtaRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GctJetCounterNegativeEtaRcd> {};
class L1GctJetCounterNegativeEtaRcd : public edm::eventsetup::DependentRecordImplementation<L1GctJetCounterNegativeEtaRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
