#include "CondTools/SiStrip/plugins/SiStripFedCablingReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripFedCablingReader::SiStripFedCablingReader( const edm::ParameterSet& pset ) :
  printFecCabling_( pset.getUntrackedParameter<bool>("PrintFecCabling",false) ),
  printDetCabling_( pset.getUntrackedParameter<bool>("PrintDetCabling",false) )
{;}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingReader::beginRun( const edm::Run& run, 
					const edm::EventSetup& setup ) {
  
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling ); 

  SiStripFecCabling* fec = new SiStripFecCabling( *cabling );
  SiStripDetCabling* det = new SiStripDetCabling( *cabling );
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    cabling->print( ss );
    ss << std::endl;
    if ( printFecCabling_ ) { fec->print( ss ); }
    ss << std::endl;
    if ( printDetCabling_ ) { det->print( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    cabling->terse( ss );
    ss << std::endl;
    if ( printFecCabling_ ) { fec->terse( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    cabling->summary( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
}
