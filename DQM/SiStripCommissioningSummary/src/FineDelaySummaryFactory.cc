#include "DQM/SiStripCommissioningSummary/interface/FineDelaySummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<FineDelayAnalysis>::SummaryHistogramFactory() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{
} 


// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<FineDelayAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<FineDelayAnalysis>::init(const sistrip::Monitorable& mon, 
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
  if ( generator_ ) { delete generator_; generator_ = 0; }
  generator_ = SummaryGenerator::instance( view );
  
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<FineDelayAnalysis>::extract( const std::map<uint32_t,FineDelayAnalysis>& data ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
         << " No data to histogram!";
    return 0; 
  } 
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
         << " NULL pointer to SummaryGenerator object!";
    return 0;  
  }

  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  std::map<uint32_t,FineDelayAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); ++iter ) {
    if ( mon_ == sistrip::FINE_DELAY_POS ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.maximum() ); 
    } else if ( mon_ == sistrip::FINE_DELAY_ERROR ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.error() ); 
    } else { continue; }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<FineDelayAnalysis>::fill( TH1& summary_histo ) {

  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
         << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
         << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if std::map is filled
  if ( !generator_->size() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
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
    edm::LogWarning(mlSummaryPlots_) << " Unexpected SummaryType value:"
         << SiStripEnumsAndStrings::presentation( pres_ );
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::FINE_DELAY_POS ) {
    generator_->axisLabel( "Delay [ns]" );
  } else if ( mon_ == sistrip::FINE_DELAY_ERROR ) { 
    generator_->axisLabel( "Uncertainty [ns]" );
  } else { 
    edm::LogWarning(mlSummaryPlots_) <<  " Unexpected SummaryHisto value:"
         << SiStripEnumsAndStrings::monitorable( mon_ ) ;
  } 
  generator_->format( sistrip::FINE_DELAY, mon_, pres_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<FineDelayAnalysis>;

