// Last commit: $Id: test_FedCablingBuilder.cc,v 1.4 2007/03/28 10:30:14 bainbrid Exp $
// Latest tag:  $Name: V02-00-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.cc,v $

#include "OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
void test_FedCablingBuilder::beginJob( const edm::EventSetup& setup ) {
  
  LogTrace(mlCabling_) 
    << "[test_FedCablingBuilder::" << __func__ << "]"
    << " Dumping all FED connections...";
  
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 
  
}

