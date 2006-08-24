#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::generate( const sistrip::SummaryHisto& histo, 
					   const sistrip::SummaryType& type,
					   const sistrip::View& view, 
					   const string& directory, 
					   const map<uint32_t,T>& data,
					   TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }
  
  // Transfer appropriate info from monitorables map to generator object
  typename map<uint32_t,T>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    generator->fillMap( directory, iter->first, (float)iter->second ); // no error here added by default 
  }

  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_SIMPLE_DISTR ) {
    generator->simpleDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_LOGICAL_VIEW ) {
    generator->logicalView( summary_histo );
  } else { return; }
  
  // Histogram formatting
  generator->format( histo, type, view, directory, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<uint32_t>;
template class SummaryHistogramFactory<uint16_t>;
template class SummaryHistogramFactory<float>;

