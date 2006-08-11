#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
string SummaryHistogramFactory<ApvTimingAnalysis::Monitorables>::name( const sistrip::SummaryHisto& histo, 
								       const sistrip::SummaryType& type,
								       const sistrip::View& view, 
								       const string& directory ) {
  
  stringstream ss;
  ss << sistrip::summaryHisto_ << sistrip::sep_;
  if ( histo == sistrip::APV_TIMING_COARSE ) {
    ss << sistrip::apvTimingCoarse_;
  } else if ( histo == sistrip::APV_TIMING_FINE ) { 
    ss << sistrip::apvTimingFine_;
  } else if ( histo == sistrip::APV_TIMING_DELAY ) { 
    ss << sistrip::apvTimingDelay_;
  } else if ( histo == sistrip::APV_TIMING_ERROR ) { 
    ss << sistrip::apvTimingError_;
  } else if ( histo == sistrip::APV_TIMING_BASE ) { 
    ss << sistrip::apvTimingBase_;
  } else if ( histo == sistrip::APV_TIMING_PEAK ) { 
    ss << sistrip::apvTimingPeak_;
  } else if ( histo == sistrip::APV_TIMING_HEIGHT ) {
    ss << sistrip::apvTimingHeight_;
  } else { 
    ss << sistrip::unknownSummaryHisto_;
  } 
  ss << sistrip::sep_ << SiStripHistoNamingScheme::view( view );
  return ss.str(); 
  
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<ApvTimingAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									 const sistrip::SummaryType& type,
									 const sistrip::View& view, 
									 const string& directory, 
									 const map<uint32_t,ApvTimingAnalysis::Monitorables>& data,
									 TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate data from map to generator object
  map<uint32_t,ApvTimingAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::APV_TIMING_COARSE ) {
      generator->fillMap( directory, iter->first, iter->second.coarse_ ); 
    } else if ( histo == sistrip::APV_TIMING_FINE ) { 
      generator->fillMap( directory, iter->first, iter->second.fine_ ); 
    } else if ( histo == sistrip::APV_TIMING_DELAY ) { 
      generator->fillMap( directory, iter->first, iter->second.delay_ ); 
    } else if ( histo == sistrip::APV_TIMING_ERROR ) { 
      generator->fillMap( directory, iter->first, iter->second.error_ ); 
    } else if ( histo == sistrip::APV_TIMING_BASE ) { 
      generator->fillMap( directory, iter->first, iter->second.base_ ); 
    } else if ( histo == sistrip::APV_TIMING_PEAK ) { 
      generator->fillMap( directory, iter->first, iter->second.peak_ ); 
    } else if ( histo == sistrip::APV_TIMING_HEIGHT ) {
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
  summary_histo.SetName( name( histo, type, view, directory ).c_str() );
  summary_histo.SetTitle( name( histo, type, view, directory ).c_str() );
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

