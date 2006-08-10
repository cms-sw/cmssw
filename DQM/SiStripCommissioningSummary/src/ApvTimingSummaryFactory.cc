#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"
#include <iostream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<ApvTimingAnalysis::Monitorables>::fill( const sistrip::SummaryHisto& histo, 
								     const sistrip::View& view, 
								     const uint32_t& key,
								     const map<uint32_t,ApvTimingAnalysis::Monitorables>& data,
								     auto_ptr<SummaryGenerator> generator ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Transfer appropriate data from map to generator object
  map<uint32_t,ApvTimingAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::APV_TIMING_COARSE ) {
      generator->fillMap( key, iter->first, iter->second.coarse_ ); 
    } else if ( histo == sistrip::APV_TIMING_FINE ) { 
      generator->fillMap( key, iter->first, iter->second.fine_ ); 
    } else if ( histo == sistrip::APV_TIMING_DELAY ) { 
      generator->fillMap( key, iter->first, iter->second.delay_ ); 
    } else if ( histo == sistrip::APV_TIMING_ERROR ) { 
      generator->fillMap( key, iter->first, iter->second.error_ ); 
    } else if ( histo == sistrip::APV_TIMING_BASE ) { 
      generator->fillMap( key, iter->first, iter->second.base_ ); 
    } else if ( histo == sistrip::APV_TIMING_PEAK ) { 
      generator->fillMap( key, iter->first, iter->second.peak_ ); 
    } else if ( histo == sistrip::APV_TIMING_HEIGHT ) {
      generator->fillMap( key, iter->first, iter->second.height_ ); 
    } else { return; } 
  }
  
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<ApvTimingAnalysis::Monitorables>::format( const sistrip::SummaryHisto& histo, 
								       const sistrip::SummaryType& type,
								       const sistrip::View& view, 
								       const uint32_t& key,
								       TH1& summary ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Histogram formatting
  if ( histo == sistrip::APV_TIMING_COARSE ) {
  } else if ( histo == sistrip::APV_TIMING_FINE ) { 
  } else if ( histo == sistrip::APV_TIMING_DELAY ) { 
  } else if ( histo == sistrip::APV_TIMING_ERROR ) { 
  } else if ( histo == sistrip::APV_TIMING_BASE ) { 
  } else if ( histo == sistrip::APV_TIMING_PEAK ) { 
  } else if ( histo == sistrip::APV_TIMING_HEIGHT ) {
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<ApvTimingAnalysis::Monitorables>;

