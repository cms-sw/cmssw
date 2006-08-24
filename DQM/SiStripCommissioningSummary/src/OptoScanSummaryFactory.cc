#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<OptoScanAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									const sistrip::SummaryType& type,
									const sistrip::View& view, 
									const string& directory, 
									const map<uint32_t,OptoScanAnalysis::Monitorables>& data,
									TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate info from monitorables map to generator object
  map<uint32_t,OptoScanAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::OPTO_SCAN_LLD_BIAS ) {
      generator->fillMap( directory, iter->first, iter->second.lldBias_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_LLD_GAIN ) { 
      generator->fillMap( directory, iter->first, iter->second.lldGain_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_GAIN ) { 
      generator->fillMap( directory, iter->first, iter->second.gain_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_ERROR ) { 
      generator->fillMap( directory, iter->first, iter->second.error_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_BASE ) { 
      generator->fillMap( directory, iter->first, iter->second.base_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_PEAK ) { 
      generator->fillMap( directory, iter->first, iter->second.peak_ ); 
    } else if ( histo == sistrip::OPTO_SCAN_HEIGHT ) {
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
  if ( histo == sistrip::OPTO_SCAN_LLD_BIAS ) {
  } else if ( histo == sistrip::OPTO_SCAN_LLD_GAIN ) { 
  } else if ( histo == sistrip::OPTO_SCAN_GAIN ) { 
  } else if ( histo == sistrip::OPTO_SCAN_ERROR ) { 
  } else if ( histo == sistrip::OPTO_SCAN_BASE ) { 
  } else if ( histo == sistrip::OPTO_SCAN_PEAK ) { 
  } else if ( histo == sistrip::OPTO_SCAN_HEIGHT ) {
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<OptoScanAnalysis::Monitorables>;

