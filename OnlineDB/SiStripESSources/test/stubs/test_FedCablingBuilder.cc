// Last commit: $Id: test_FedCablingBuilder.cc,v 1.7 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.cc,v $

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
void test_FedCablingBuilder::analyze( const edm::Event& event, const edm::EventSetup& setup ) {
  
  LogTrace(mlCabling_) 
    << "[test_FedCablingBuilder::" << __func__ << "]"
    << " Dumping all FED connections...";
  
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 
  
}

