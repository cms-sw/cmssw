#ifndef DataRecord_L1RPCConfigRcd_h
#define DataRecord_L1RPCConfigRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1RPCConfigRcd
// 
/**\class L1RPCConfigRcd L1RPCConfigRcd.h CondFormats/DataRecord/interface/L1RPCConfigRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Tue Mar 20 14:39:09 CET 2007
// $Id: L1RPCConfigRcd.h,v 1.2 2008/03/03 07:09:47 wsun Exp $
//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"

//class L1RPCConfigRcd : public edm::eventsetup::EventSetupRecordImplementation<L1RPCConfigRcd> {};
class L1RPCConfigRcd : public edm::eventsetup::DependentRecordImplementation<L1RPCConfigRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};

#endif
