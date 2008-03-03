#ifndef DataRecord_L1GctJetCounterPositiveEtaRcd_h
#define DataRecord_L1GctJetCounterPositiveEtaRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1GctJetCounterPositiveEtaRcd
// 
/**\class L1GctJetCounterPositiveEtaRcd L1GctJetCounterPositiveEtaRcd.h CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Tue Jul 10 10:14:03 CEST 2007
// $Id: L1GctJetCounterPositiveEtaRcd.h,v 1.1 2007/09/17 11:22:33 heath Exp $
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1GctJetCounterPositiveEtaRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GctJetCounterPositiveEtaRcd> {};
class L1GctJetCounterPositiveEtaRcd : public edm::eventsetup::DependentRecordImplementation<L1GctJetCounterPositiveEtaRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
