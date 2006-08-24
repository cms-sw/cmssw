#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
string SummaryHistogramFactory<VpspScanAnalysis::Monitorables>::name( const sistrip::SummaryHisto& histo, 
								      const sistrip::SummaryType& type,
								      const sistrip::View& view, 
								      const string& directory ) {
  
  stringstream ss;
  ss << sistrip::summaryHisto_ << sistrip::sep_;
  if ( histo == sistrip::VPSP_SCAN_APV0 ) {
    ss << sistrip::vpspScanApv0_;
  } else if ( histo == sistrip::VPSP_SCAN_APV0 ) { 
    ss << sistrip::vpspScanApv1_;
  } else { 
    ss << sistrip::unknownSummaryHisto_;
  } 
  ss << sistrip::sep_ << SiStripHistoNamingScheme::view( view );
  return ss.str(); 
  
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<VpspScanAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									const sistrip::SummaryType& type,
									const sistrip::View& view, 
									const string& directory, 
									const map<uint32_t,VpspScanAnalysis::Monitorables>& data,
									TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate info from monitorables map to generator object
  map<uint32_t,VpspScanAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::VPSP_SCAN_APV0 ) {
      generator->fillMap( directory, iter->first, iter->second.vpsp0_ ); 
    } else if ( histo == sistrip::VPSP_SCAN_APV1 ) {
      generator->fillMap( directory, iter->first, iter->second.vpsp0_ ); 
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
  if ( histo == sistrip::VPSP_SCAN_APV0 ) {
  } else if ( histo == sistrip::VPSP_SCAN_APV1 ) {
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<VpspScanAnalysis::Monitorables>;

