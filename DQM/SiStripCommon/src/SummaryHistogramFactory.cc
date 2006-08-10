#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SummaryGeneratorControlView.h"
//#include "DQM/SiStripCommon/interface/SummaryGeneratorReadoutView.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::generate( const sistrip::SummaryHisto& histo, 
					   const sistrip::SummaryType& type,
					   const sistrip::View& view, 
					   const string& directory, 
					   const map<uint32_t,T>& data,
					   TH1& summary ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Create utility class that generates summary histogram  
  auto_ptr<SummaryGenerator> generator;
  if ( view == sistrip::CONTROL ) {
    generator = auto_ptr<SummaryGenerator>( new SummaryGeneratorControlView() );
  } else if ( view == sistrip::READOUT ) {
    //generator = auto_ptr<SummaryGenerator>( new SummaryGeneratorReadoutView() );
  } else { return; }
  
  // Extract key from directory string that represents histogramming level
  uint32_t key = 0;
  if ( view == sistrip::CONTROL ) {
    SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath( directory );
    key = SiStripControlKey::key( path.fecCrate_,
				  path.fecSlot_,
				  path.fecRing_,
				  path.ccuAddr_,
				  path.ccuChan_ );
  } else if ( view == sistrip::READOUT ) {
    std::pair<uint16_t,uint16_t> path = SiStripHistoNamingScheme::readoutPath( directory );
    key = SiStripReadoutKey::key( path.first,
				  path.second );
  } else { return; }
  
  // Extract parameter to be histogrammed from map
  fill( histo, view, key, data, generator );

  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_SIMPLE_DISTR ) {
    generator->simpleDistr( summary );
  } else if ( type == sistrip::SUMMARY_LOGICAL_VIEW ) {
    generator->logicalView( summary );
  } else { return; }
  
  // Format histogram (if necessary)
  format( histo, type, view, key, summary );
  
}

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::fill( const sistrip::SummaryHisto& histo, 
				       const sistrip::View& view, 
				       const uint32_t& key,
				       const map<uint32_t,T>& data,
				       auto_ptr<SummaryGenerator> generator ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Transfer appropriate data from map to generator object
  typename map<uint32_t,T>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    generator->fillMap( key, iter->first, (float)iter->second ); // no error here added by default 
  }
  
}

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::format( const sistrip::SummaryHisto& histo, 
					 const sistrip::SummaryType& type,
					 const sistrip::View& view, 
					 const uint32_t& key,
					 TH1& summary ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<uint32_t>;
template class SummaryHistogramFactory<uint16_t>;
template class SummaryHistogramFactory<float>;

