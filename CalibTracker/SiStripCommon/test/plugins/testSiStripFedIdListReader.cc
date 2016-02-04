#include "CalibTracker/SiStripCommon/test/plugins/testSiStripFedIdListReader.h"
#include "CalibTracker/SiStripCommon/interface/SiStripFedIdListReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
// -----------------------------------------------------------------------------
//
testSiStripFedIdListReader::testSiStripFedIdListReader( const edm::ParameterSet& pset ) 
  : fileInPath_( pset.getParameter<edm::FileInPath>("file") )
{
  edm::LogVerbatim("Unknown") 
    << "[testSiStripFedIdListReader::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void testSiStripFedIdListReader::analyze( const edm::Event& event, 
					  const edm::EventSetup& setup ) {
  
  SiStripFedIdListReader reader( fileInPath_.fullPath() );
  edm::LogVerbatim("Unknown") 
    << "[testSiStripFedIdListReader::" << __func__ << "]"
    << reader;
}

