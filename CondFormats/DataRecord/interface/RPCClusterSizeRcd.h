#ifndef DataRecord_RPCClusterSizeRcd_h
#define DataRecord_RPCClusterSizeRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     RPCClusterSizeRcd
// 
/**\class RPCClusterSizeRcd RPCClusterSizeRcd.h CondFormats/DataRecord/interface/RPCClusterSizeRcd.h

 Description:  Record for the cluster size for each RPC  

 Usage:
    used by SimMuon/RPCDigitizer 

*/
//
// Author:     Borislav Pavlov 
// Created:     Mon Nov  2 17:43:16 CET 2009
// $Id: RPCClusterSizeRcd.h,v 1.1 2009/12/10 13:05:02 futyand Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class RPCClusterSizeRcd : public edm::eventsetup::EventSetupRecordImplementation<RPCClusterSizeRcd> {};

#endif
