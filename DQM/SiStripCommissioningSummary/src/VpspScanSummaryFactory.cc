#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<VpspScanAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									const sistrip::SummaryType& type,
									const sistrip::View& view, 
									const string& directory, 
									const map<uint32_t,VpspScanAnalysis::Monitorables>& data,
									TH1& summary_histo ) {

  // Check if data are present
  if ( data.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data to histogram!" << endl;
    return ; 
  } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate info from monitorables map to generator object
  map<uint32_t,VpspScanAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::VPSP_SCAN_APV0 ) {
      generator->fillMap( directory, iter->first, iter->second.vpsp0_ ); 
    } else if ( histo == sistrip::VPSP_SCAN_APV1 ) {
      generator->fillMap( directory, iter->first, iter->second.vpsp1_ ); 
    } else { return; } 
  }
  
  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_DISTR ) {
    generator->summaryDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_1D ) {
    generator->summary1D( summary_histo );
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

