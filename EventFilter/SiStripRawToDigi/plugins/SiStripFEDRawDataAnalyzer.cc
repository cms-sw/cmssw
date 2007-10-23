#include "EventFilter/SiStripRawToDigi/plugins/SiStripFEDRawDataAnalyzer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDRawDataCheck.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataAnalyzer::SiStripFEDRawDataAnalyzer( const edm::ParameterSet& pset ) 
  : check_( new SiStripFEDRawDataCheck(pset) )
{
  LogTrace(mlRawToDigi_)
    << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataAnalyzer::~SiStripFEDRawDataAnalyzer() 
{
  LogTrace(mlRawToDigi_)
    << "[SiStripFEDRawDataAnalyzer::" << __func__ << "]"
    << " Destructing object...";
  if ( check_ ) { delete check_; }
}

// -----------------------------------------------------------------------------
// 
void SiStripFEDRawDataAnalyzer::analyze( const edm::Event& event,
					 const edm::EventSetup& setup ) {
  if ( check_ ) { check_->analyze( event, setup ); }
}
