#include "DQM/SiStripCommissioningSummary/interface/FedTimingSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedTimingAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									 const sistrip::SummaryType& type,
									 const sistrip::View& view, 
									 const string& directory, 
									 const map<uint32_t,FedTimingAnalysis::Monitorables>& data,
									 TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate info from monitorables map to generator object
  map<uint32_t,FedTimingAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::FED_TIMING_PLL_COARSE ) {
      generator->fillMap( directory, iter->first, iter->second.pllCoarse_ ); 
    } else if ( histo == sistrip::FED_TIMING_PLL_FINE ) { 
      generator->fillMap( directory, iter->first, iter->second.pllFine_ ); 
    } else if ( histo == sistrip::FED_TIMING_DELAY ) { 
      generator->fillMap( directory, iter->first, iter->second.delay_ ); 
    } else if ( histo == sistrip::FED_TIMING_ERROR ) { 
      generator->fillMap( directory, iter->first, iter->second.error_ ); 
    } else if ( histo == sistrip::FED_TIMING_BASE ) { 
      generator->fillMap( directory, iter->first, iter->second.base_ ); 
    } else if ( histo == sistrip::FED_TIMING_PEAK ) { 
      generator->fillMap( directory, iter->first, iter->second.peak_ ); 
    } else if ( histo == sistrip::FED_TIMING_HEIGHT ) {
      generator->fillMap( directory, iter->first, iter->second.height_ ); 
    } else { return; } 
  }
  
  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_SIMPLE_DISTR ) {
    generator->simpleDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_LOGICAL_VIEW ) {
    generator->logicalView( summary_histo );
  } else { return; }

  // Histogram formatting
  generator->format( histo, type, view, directory, summary_histo );
//   summary_histo.SetName( name( histo, type, view, directory ).c_str() );
//   summary_histo.SetTitle( name( histo, type, view, directory ).c_str() );
//   generator->format( summary_histo );
  if ( histo == sistrip::FED_TIMING_PLL_COARSE ) {
  } else if ( histo == sistrip::FED_TIMING_PLL_FINE ) { 
  } else if ( histo == sistrip::FED_TIMING_DELAY ) { 
  } else if ( histo == sistrip::FED_TIMING_ERROR ) { 
  } else if ( histo == sistrip::FED_TIMING_BASE ) { 
  } else if ( histo == sistrip::FED_TIMING_PEAK ) { 
  } else if ( histo == sistrip::FED_TIMING_HEIGHT ) {
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<FedTimingAnalysis::Monitorables>;

