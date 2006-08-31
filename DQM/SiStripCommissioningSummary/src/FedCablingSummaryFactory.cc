#include "DQM/SiStripCommissioningSummary/interface/FedCablingSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedCablingAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									  const sistrip::SummaryType& type,
									  const sistrip::View& view, 
									  const string& directory, 
									  const map<uint32_t,FedCablingAnalysis::Monitorables>& data,
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
  map<uint32_t,FedCablingAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::FED_CABLING_FED_ID ) {
      generator->fillMap( directory, iter->first, iter->second.fedId_ ); 
    } else if ( histo == sistrip::FED_CABLING_FED_ID ) { 
      generator->fillMap( directory, iter->first, iter->second.fedCh_ ); 
    } else { return; } 
  }
  
  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_DISTR ) {
    generator->summaryDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_1D ) {
    generator->summary1D( summary_histo );
  } else if ( type == sistrip::SUMMARY_2D ) {
    generator->summary2D( summary_histo );
  } else { return; }

  // Histogram formatting
  generator->format( histo, type, view, directory, summary_histo );
  if ( histo == sistrip::FED_CABLING_FED_ID ) {
  } else if ( histo == sistrip::FED_CABLING_FED_ID ) { 
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<FedCablingAnalysis::Monitorables>;

