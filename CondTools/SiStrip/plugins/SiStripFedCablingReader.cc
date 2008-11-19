#include "CondTools/SiStrip/plugins/SiStripFedCablingReader.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingReader::beginRun( const edm::Run& run, 
				      const edm::EventSetup& setup ) {
  
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling ); 
  
  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    cabling->print( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    cabling->terse( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    cabling->summary( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  
}
