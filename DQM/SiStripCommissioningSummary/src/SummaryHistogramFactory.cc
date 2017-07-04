#include "DQM/SiStripCommissioningSummary/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
template<class T> 
SummaryHistogramFactory<T>::SummaryHistogramFactory() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(nullptr) 
{
  LogTrace(mlSummaryPlots_) 
    << "[SummaryHistogramFactory::" << __func__ << "]";
} 

// -----------------------------------------------------------------------------
//
template<class T> 
SummaryHistogramFactory<T>::~SummaryHistogramFactory() {
  LogTrace(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]";
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
template<class T> 
void SummaryHistogramFactory<T>::init( const sistrip::Monitorable& mon, 
				       const sistrip::Presentation& pres,
				       const sistrip::View& view, 
				       const std::string& top_level_dir, 
				       const sistrip::Granularity& gran ) {
  LogTrace(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]";
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { 
    delete generator_;
    generator_ = nullptr;
    generator_ = SummaryGenerator::instance( view );
  }
  
}

// -----------------------------------------------------------------------------
//
template<class T> 
uint32_t SummaryHistogramFactory<T>::extract( const std::map<uint32_t,T>& data ) {
  LogTrace(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]";
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
				     << " No data in monitorables std::map!";
    return 0; 
  }
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return 0;
  }
  
  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  typename std::map<uint32_t,T>::const_iterator iter = data.begin();
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
  LogTrace(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]";
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if std::map is filled
  if ( !generator_->size() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " No data in the monitorables std::map!";
    return; 
  } 
  
  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::HISTO_1D ) {
    generator_->histo1D( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SUM ) {
    generator_->histo2DSum( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SCATTER ) {
    generator_->histo2DScatter( summary_histo );
  } else if ( pres_ == sistrip::PROFILE_1D ) {
    generator_->profile1D( summary_histo );
  } else { return; }
  
  // Histogram formatting
  generator_->format( sistrip::UNKNOWN_RUN_TYPE, mon_, pres_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<uint32_t>;
template class SummaryHistogramFactory<uint16_t>;
template class SummaryHistogramFactory<float>;

