#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<ApvTimingAnalysis*>::init( const sistrip::Monitorable& mon, 
						       const sistrip::Presentation& pres,
						       const sistrip::View& view, 
						       const std::string& level, 
						       const sistrip::Granularity& gran,
						       const std::map<uint32_t,ApvTimingAnalysis*>& data ) {
  
  // Check if generator class exists
  if ( !generator_ ) { return 0; }

  // Check if generator object exists
  //if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Extract monitorable and fill map
  std::map<uint32_t,ApvTimingAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    float value = 1. * sistrip::invalid_;
    float error = 1. * sistrip::invalid_;
    if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_TIME ) { 
      value = iter->second->time(); 
      error = iter->second->error();
    } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_MAX_TIME ) { value = iter->second->refTime(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_DELAY ) { value = iter->second->delay(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_BASE ) { value = iter->second->base(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_PEAK ) { value = iter->second->peak(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_HEIGHT ) { value = iter->second->height(); }
    else { 
      edm::LogWarning(mlSummaryPlots_)
	<< "[SummaryPlotFactory::" << __func__ << "]" 
	<< " Unexpected monitorable: "
	<< SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
      continue; 
    }

    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value );
  }
  
  return SummaryPlotFactoryBase::generator_->nBins();
  
}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<ApvTimingAnalysis*>::fill( TH1& summary_histo ) {

  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );

  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }

  // Histogram formatting
  if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_TIME ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "Timing delay [ns]" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_MAX_TIME ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_DELAY ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_ERROR ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_BASE ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_PEAK ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_HEIGHT ) {
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<ApvTimingAnalysis*>;

