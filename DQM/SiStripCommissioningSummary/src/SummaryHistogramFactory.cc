#include "DQM/SiStripCommissioningSummary/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
template<class T> 
SummaryHistogramFactory<T>::SummaryHistogramFactory() :
  histo_(sistrip::UNKNOWN_SUMMARY_HISTO),
  type_(sistrip::UNKNOWN_SUMMARY_TYPE),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
} 

// -----------------------------------------------------------------------------
//
template<class T> 
SummaryHistogramFactory<T>::~SummaryHistogramFactory() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::init( const sistrip::SummaryHisto& histo, 
				       const sistrip::SummaryType& type,
				       const sistrip::View& view, 
				       const string& top_level_dir, 
				       const sistrip::Granularity& gran ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  histo_ = histo;
  type_ = type;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { 
    delete generator_;
    generator_ = 0;
    generator_ = SummaryGenerator::instance( view );
  }
  
}

// -----------------------------------------------------------------------------
//
template<class T> 
uint32_t SummaryHistogramFactory<T>::extract( const map<uint32_t,T>& data ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Check if data are present
  if ( data.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data in monitorables map!" << endl;
    return 0; 
  }
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return 0;
  }
  
  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  typename map<uint32_t,T>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    generator_->fillMap( level_,                             // top-level directory
			 gran_,                              // granularity
			 iter->first,                        // device key
			 static_cast<float>(iter->second) ); // value
  }
  
  return generator_->size();
}

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::fill( TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }

  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }

  // Check if map is filled
  if ( !generator_->size() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data in the monitorables map!" << endl;
    return; 
  } 
  
  // Generate appropriate summary histogram 
  if ( type_ == sistrip::SUMMARY_DISTR ) {
    generator_->summaryDistr( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_1D ) {
    generator_->summary1D( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_2D ) {
    generator_->summary2D( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_PROF ) {
    generator_->summaryProf( summary_histo );
  } else { return; }
  
  // Histogram formatting
  generator_->format( sistrip::UNKNOWN_TASK, histo_, type_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<uint32_t>;
template class SummaryHistogramFactory<uint16_t>;
template class SummaryHistogramFactory<float>;

