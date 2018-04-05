#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SummaryGeneratorControlView::SummaryGeneratorControlView() :
  SummaryGenerator("SummaryGeneratorControlView") {;}

// -----------------------------------------------------------------------------
// 
void SummaryGeneratorControlView::fill( const std::string& top_level_dir,
					const sistrip::Granularity& granularity,
					const uint32_t& device_key, 
					const float& value,
					const float& error ) {
  
  // Check granularity is recognised
  std::string gran = SiStripEnumsAndStrings::granularity( granularity );
  
  if ( granularity != sistrip::UNDEFINED_GRAN &&
       granularity != sistrip::FEC_CRATE &&
       granularity != sistrip::FEC_SLOT &&
       granularity != sistrip::FEC_RING &&
       granularity != sistrip::CCU_ADDR &&
       granularity != sistrip::CCU_CHAN &&
       granularity != sistrip::LLD_CHAN && 
       granularity != sistrip::APV ) {
    std::string temp = SiStripEnumsAndStrings::granularity( sistrip::LLD_CHAN );
    edm::LogWarning(mlSummaryPlots_) 
      << "[SummaryGeneratorControlView::" << __func__ << "]"
      << " Unexpected granularity requested: " << gran;
    return;
  }
  
  // Create key representing "top level" directory 
  SiStripFecKey top( top_level_dir );
  
  // Path and std::string for "present working directory" as defined by device key
  SiStripFecKey path( device_key );
  const std::string& pwd = path.path();

//   LogTrace(mlTest_)
//     << "TEST " 
//     << "top " << top 
//     << "path " << path;
  
  if ( top.isValid() &&
       path.isValid() &&
       ( path.fecCrate() == top.fecCrate() || !top.fecCrate() ) && 
       ( path.fecSlot() == top.fecSlot() || !top.fecSlot() ) && 
       ( path.fecRing() == top.fecRing() || !top.fecRing() ) && 
       ( path.ccuAddr() == top.ccuAddr() || !top.ccuAddr() ) && 
       ( path.ccuChan() == top.ccuChan() || !top.ccuChan() ) ) {
    
    // Extract path and std::string corresponding to "top-level down to granularity" 
    std::string sub_dir = pwd;
    size_t pos = pwd.find( gran );
    if ( pos != std::string::npos ) {
      sub_dir = pwd.substr( 0, pwd.find(sistrip::dir_,pos) );
    } else if ( granularity == sistrip::UNKNOWN_GRAN ) {
      sub_dir = pwd;
    }

    SiStripFecKey sub_path( sub_dir );

//     LogTrace(mlTest_)
//       << "TEST " 
//       << "sub_path " << sub_path; 
    
    // Construct bin label
    std::stringstream bin;
    if ( sub_path.fecCrate() != sistrip::invalid_ ) { bin << std::setw(1) << std::setfill('0') << sub_path.fecCrate(); }
    if ( sub_path.fecSlot()  != sistrip::invalid_ ) { bin << sistrip::dot_ << std::setw(2) << std::setfill('0') << sub_path.fecSlot(); }
    if ( sub_path.fecRing()  != sistrip::invalid_ ) { bin << sistrip::dot_ << std::setw(1) << std::setfill('0') << sub_path.fecRing(); }
    if ( sub_path.ccuAddr()  != sistrip::invalid_ ) { bin << sistrip::dot_ << std::setw(3) << std::setfill('0') << sub_path.ccuAddr(); }
    if ( sub_path.ccuChan()  != sistrip::invalid_ ) { bin << sistrip::dot_ << std::setw(2) << std::setfill('0') << sub_path.ccuChan(); }
    if ( ( granularity == sistrip::LLD_CHAN || 
	   granularity == sistrip::APV ) && 
	 path.channel() != sistrip::invalid_ ) { bin << sistrip::dot_ << path.channel(); }
    
    // Store "value" in appropriate std::vector within std::map (key is bin label)
    map_[bin.str()].push_back( Data(value,error) );
    entries_++;
//     LogTrace(mlTest_)
//       << "TEST " 
//       << " filling " << bin.str() << " " << value << " " << error << " ";

  }
  
}

