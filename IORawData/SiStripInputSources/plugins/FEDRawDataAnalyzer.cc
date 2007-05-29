#include "IORawData/SiStripInputSources/plugins/FEDRawDataAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "interface/shared/include/fed_trailer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unistd.h>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
FEDRawDataAnalyzer::FEDRawDataAnalyzer( const edm::ParameterSet& pset ) 
  : label_( pset.getUntrackedParameter<std::string>("ProductLabel","source") ),
    instance_( pset.getUntrackedParameter<std::string>("ProductInstance","") ),
    sleep_( pset.getUntrackedParameter<int>("pause_us",0) )
{
  LogTrace(mlRawToDigi_)
    << "[FEDRawDataAnalyzer::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::analyze( const edm::Event& event,
				  const edm::EventSetup& setup ) {
  
  // Introduce optional delay
  if ( sleep_ > 0 ) { for ( int ii = 0; ii < sleep_; ++ii ) { usleep(1); } }
  
  // Retrieve FED raw data
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( label_, instance_, buffers ); 
  
  // Some init
  std::stringstream ss; 
  ss << std::endl;
  uint16_t cntr = 0;
  bool trigger_fed = false;
  
  // Check FEDRawDataCollection
  for ( uint16_t ifed = 0; ifed < sistrip::CMS_FED_ID_MAX; ifed++ ) {
    const FEDRawData& fed = buffers->FEDData( static_cast<int>(ifed) );
    if ( fed.size() ) { 
      cntr++;
      if ( edm::isDebugEnabled() ) {
	ss << " #FEDs: " 
	   << std::setw(3) << std::setfill(' ') << cntr
	   << "  FED id: " 
	   << std::setw(4) << std::setfill(' ') << ifed 
	   << "  Buffer size [chars]: " 
	   << std::setw(6) << std::setfill(' ') << fed.size();
      }
      uint8_t* data = const_cast<uint8_t*>( fed.data() );
      fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( data + fed.size() - sizeof(fedt_t) );
      if ( fed_trailer->conscheck == 0xDEADFACE ) { 
	ss << " (Candidate for \"trigger FED\"!)";
	trigger_fed = true; 
      } 
      ss << std::endl;
    }
  }
  
  // Print out debug
  std::stringstream sss;
  sss << "[FEDRawDataAnalyzer::" << __func__ << "]"
      << " Number of FED buffers (with non-zero size) found in collection is "
      << cntr;
  if ( trigger_fed ) { sss << " (One buffer contains the \"trigger FED\" info!)"; }
  edm::LogVerbatim(mlRawToDigi_) << sss.str();
  
  LogTrace(mlRawToDigi_) 
    << "[FEDRawDataAnalyzer::" << __func__ << "]"
    << " Buffer size for " << cntr << " FEDs: "
    << ss.str();
  
}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::beginJob( edm::EventSetup const& ) {;}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::endJob() {;}
