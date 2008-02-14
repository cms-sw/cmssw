#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void ApvTimingSummaryFactory::extract( Iterator iter ) {
  
  ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>( iter->second );
  if ( !anal ) { return; }
    
  float value = 1. * sistrip::invalid_;
  
  if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_TIME ) { 
    value = anal->time(); 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_MAX_TIME ) { 
    value = anal->refTime(); 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_DELAY ) { 
    value = anal->delay(); 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_BASE ) { 
    value = anal->base(); 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_PEAK ) { 
    value = anal->peak(); 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_HEIGHT ) { 
    value = anal->height(); 
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected monitorable: "
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
    return; 
  }
  
  SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
					       SummaryPlotFactoryBase::gran_, 
					       iter->first, 
					       value );
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingSummaryFactory::format() {
  
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
