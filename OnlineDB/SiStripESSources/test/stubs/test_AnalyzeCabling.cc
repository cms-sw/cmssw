// Last commit: $Id: test_AnalyzeCabling.cc,v 1.2 2008/06/05 14:59:15 bainbrid Exp $

#include "OnlineDB/SiStripESSources/test/stubs/test_AnalyzeCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
void test_AnalyzeCabling::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  
  using namespace sistrip;
  
  // fed cabling
  LogTrace(mlCabling_) 
    << "[test_AnalyzeCabling::" << __func__ << "]"
    << " Dumping all connection objects in FED cabling..."
    << std::endl;
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 
  
  // fec cabling
  SiStripFecCabling fec_cabling( *fed_cabling );
  std::stringstream ss;
  ss << "[test_AnalyzeCabling::" << __func__ << "]"
     << " Dumping all SiStripModule objects in FEC cabling..."
     << std::endl << fec_cabling;
  LogTrace(mlCabling_) << ss.str();
  
}

