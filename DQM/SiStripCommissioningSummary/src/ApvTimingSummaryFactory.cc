#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<ApvTimingAnalysis*>::init( const sistrip::Monitorable& mon, 
						       const sistrip::Presentation& pres,
						       const sistrip::View& view, 
						       const string& level, 
						       const sistrip::Granularity& gran,
						       const map<uint32_t,ApvTimingAnalysis*>& data ) {
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Transfer appropriate monitorables info to generator object
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  map<uint32_t,ApvTimingAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    static float value = 1.*sistrip::invalid_;
    if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_TIME ) { value = iter->second->time(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_MAX_TIME ) { value = iter->second->maxTime(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_DELAY ) { value = iter->second->delay(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_ERROR ) { value = iter->second->error(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_BASE ) { value = iter->second->base(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_PEAK ) { value = iter->second->peak(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_HEIGHT ) { value = iter->second->height(); }
    else { continue; }
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

  // Hitsogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  // Histogram formatting
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }
  if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_TIME ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "Timing delay [ns]" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_MAX_TIME ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_DELAY ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_ERROR ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_BASE ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_PEAK ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::APV_TIMING_HEIGHT ) {
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " Unexpected SummaryHisto value:"
	 << SiStripHistoNamingScheme::monitorable( SummaryPlotFactoryBase::mon_ ) 
	 << endl;
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<ApvTimingAnalysis>;

