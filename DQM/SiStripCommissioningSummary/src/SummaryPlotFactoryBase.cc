#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryPlotFactoryBase::SummaryPlotFactoryBase() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{;} 

// -----------------------------------------------------------------------------
//
SummaryPlotFactoryBase::~SummaryPlotFactoryBase() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::init( const sistrip::Monitorable& mon, 
				   const sistrip::Presentation& pres,
				   const sistrip::View& view, 
				   const std::string& level, 
				   const sistrip::Granularity& gran ) {
  
  // Retrieve utility class used to generate summary histograms
  if ( view != view_ && generator_ ) { 
    delete generator_; 
    generator_ = 0;
  }
  generator_ = SummaryGenerator::instance( view );
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactoryBase::" << __func__ << "]" 
      << " NULL pointer to SummaryGenerator object!";
    return;
  }
  
//   // Check if std::map needs to be cleared
//   if ( mon != mon_ || 
//        view != view_ || 
//        level != level_ || 
//        gran != gran_ ) {
//     generator_->clearMap();
//   }

  // should always be cleared?... 
  // (what if std::map data change? yes!)
  generator_->clearMap();
  
  // Set parameters
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = level;
  gran_ = gran;

  std::stringstream ss;
  ss << "[SummaryPlotFactoryBase::" << __func__ << "]" 
     << " Dump of parameters defining summary plot:" << std::endl
     << " Monitorable   : " << SiStripEnumsAndStrings::monitorable( mon_ ) << std::endl
     << " Presentation  : " << SiStripEnumsAndStrings::presentation( pres_ ) << std::endl
     << " Logical view  : " << SiStripEnumsAndStrings::view( view_ ) << std::endl
     << " Top level dir : " << level_ << std::endl
     << " Granularity   : " << SiStripEnumsAndStrings::granularity( gran_ );
  LogTrace(mlSummaryPlots_) << ss.str();
  
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::fill( TH1& summary_histo ) {
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) 
      << "[SummaryPlotFactoryBase::" << __func__ << "]" 
      << " NULL pointer to SummaryGenerator object!";
    return;
  }
  
  // Check if map is filled
  if ( !generator_->nBins() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactoryBase::" << __func__ << "]" 
      << " SummaryGenerator::map_ is empty!";
    return; 
  } 
  
  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactoryBase::" << __func__ << "]" 
      << " NULL pointer to TH1 object!";
    return;
  }
  
  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::HISTO_1D ) {
    generator_->summaryHisto( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SUM ) {
    generator_->summary1D( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SCATTER ) {
    generator_->summary2D( summary_histo );
  } else if ( pres_ == sistrip::PROFILE_1D ) {
    generator_->summaryProf( summary_histo );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactoryBase::" << __func__ << "]" 
      << " Unexpected presentation type: "
      << SiStripEnumsAndStrings::presentation( pres_ );
    return; 
  }
  
  // Histogram formatting
  generator_->format( sistrip::UNKNOWN_RUN_TYPE, 
		      mon_, 
		      pres_, 
		      view_, 
		      level_, 
		      gran_, 
		      summary_histo );
  
}
