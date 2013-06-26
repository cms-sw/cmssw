#ifndef DataRecord_RPCObCondRcd_h
#define DataRecord_RPCObCondRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     RPCObCondRcd
// 
/**\class RPCObCondRcd RPCObCondRcd.h CondFormats/DataRecord/interface/RPCObCondRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Fri Oct 10 20:02:37 CEST 2008
// $Id: RPCObCondRcd.h,v 1.2 2008/12/12 17:30:31 dpagano Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class RPCObImonRcd : public edm::eventsetup::EventSetupRecordImplementation<RPCObImonRcd> {};

class RPCObVmonRcd : public edm::eventsetup::EventSetupRecordImplementation<RPCObVmonRcd> {};

class RPCObStatusRcd : public edm::eventsetup::EventSetupRecordImplementation<RPCObStatusRcd> {};

class RPCObTempRcd : public edm::eventsetup::EventSetupRecordImplementation<RPCObTempRcd> {};

#endif
