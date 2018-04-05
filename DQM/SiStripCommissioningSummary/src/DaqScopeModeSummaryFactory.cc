#include "DQM/SiStripCommissioningSummary/interface/DaqScopeModeSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<DaqScopeModeAnalysis>::SummaryHistogramFactory() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(nullptr) 
{;} 


// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<DaqScopeModeAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<DaqScopeModeAnalysis>::init( const sistrip::Monitorable& mon, 
							  const sistrip::Presentation& pres,
							  const sistrip::View& view, 
							  const std::string& top_level_dir, 
							  const sistrip::Granularity& gran ) {
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { delete generator_; generator_ = nullptr; }
  generator_ = SummaryGenerator::instance( view );
  
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<DaqScopeModeAnalysis>::extract( const std::map<uint32_t,DaqScopeModeAnalysis>& data ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " No data in monitorables std::map!";
    return 0; 
  }
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " NULL pointer to SummaryGenerator object!";
    return 0;
  }
  
  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  std::map<uint32_t,DaqScopeModeAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( mon_ == sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.mean() ); 
    } else { continue; }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<DaqScopeModeAnalysis>::fill( TH1& summary_histo ) {

  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if std::map is filled
  if ( !generator_->size() ) { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " No data in the monitorables std::map!";
    return; 
  } 

  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::HISTO_1D ) {
    generator_->histo1D( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SUM ) {
    generator_->histo2DSum( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SCATTER ) {
    generator_->histo2DScatter( summary_histo );
  } else if ( pres_ == sistrip::PROFILE_1D ) {
    generator_->profile1D( summary_histo );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " Unexpected SummaryType value:"
      << SiStripEnumsAndStrings::presentation( pres_ );
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL ) { 
    generator_->axisLabel( "Mean signal [adc]" );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryHistogramFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( mon_ );
  } 
  generator_->format( sistrip::APV_TIMING, mon_, pres_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<DaqScopeModeAnalysis>;

