#include "CondTools/SiStrip/plugins/SiStripFedCablingReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripFedCablingReader::SiStripFedCablingReader( const edm::ParameterSet& pset ) :
  printFecCabling_( pset.getUntrackedParameter<bool>("PrintFecCabling",false) ),
  printDetCabling_( pset.getUntrackedParameter<bool>("PrintDetCabling",false) ),
  printRegionCabling_( pset.getUntrackedParameter<bool>("PrintRegionCabling",false) )
{;}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingReader::beginRun( const edm::Run& run, 
					const edm::EventSetup& setup ) {

  edm::LogVerbatim("SiStripFedCablingReader") 
    << "[SiStripFedCablingReader::" << __func__ << "]"
    << " Retrieving FED cabling...";
  edm::ESHandle<SiStripFedCabling> fed;
  setup.get<SiStripFedCablingRcd>().get( fed ); 

  edm::LogVerbatim("SiStripFedCablingReader") 
    << "[SiStripFedCablingReader::" << __func__ << "]"
    << " Retrieving FEC cabling...";
  edm::ESHandle<SiStripFecCabling> fec;
  setup.get<SiStripFecCablingRcd>().get( fec ); 

  edm::LogVerbatim("SiStripFedCablingReader") 
    << "[SiStripFedCablingReader::" << __func__ << "]"
    << " Retrieving DET cabling...";
  edm::ESHandle<SiStripDetCabling> det;
  setup.get<SiStripDetCablingRcd>().get( det ); 

  edm::LogVerbatim("SiStripFedCablingReader") 
    << "[SiStripFedCablingReader::" << __func__ << "]"
    << " Retrieving REGION cabling...";
  edm::ESHandle<SiStripRegionCabling> region;
  setup.get<SiStripRegionCablingRcd>().get( region ); 

  if ( !fed.isValid() ) {
    edm::LogError("SiStripFedCablingReader") 
      << " Invalid handle to FED cabling object: ";
    return;
  }

  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    fed->print( ss );
    ss << std::endl;
    if ( printFecCabling_ && fec.isValid() ) { fec->print( ss ); }
    ss << std::endl;
    if ( printDetCabling_ && det.isValid() ) { det->print( ss ); }
    ss << std::endl;
    if ( printRegionCabling_ && region.isValid() ) { region->print( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    fed->summary( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
  
}
