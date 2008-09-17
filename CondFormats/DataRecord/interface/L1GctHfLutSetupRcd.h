#ifndef DataRecord_L1GctHfLutSetupRcd_h
#define DataRecord_L1GctHfLutSetupRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1GctHfLutSetupRcd
// 
/**\class L1GctHfLutSetupRcd L1GctHfLutSetupRcd.h CondFormats/DataRecord/interface/L1GctHfLutSetupRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Tue Jul 10 10:14:03 CEST 2007
// $Id: L1GctHfLutSetupRcd.h,v 1.2 2008/03/03 07:09:47 wsun Exp $
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1GctHfLutSetupRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GctHfLutSetupRcd> {};
class L1GctHfLutSetupRcd : public edm::eventsetup::DependentRecordImplementation<L1GctHfLutSetupRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
