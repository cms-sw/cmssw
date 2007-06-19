#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<VpspScanAnalysis>::SummaryHistogramFactory() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{
} 


// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<VpspScanAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<VpspScanAnalysis>::init( const sistrip::Monitorable& mon, 
						      const sistrip::Presentation& pres,
						      const sistrip::View& view, 
						      const std::string& top_level_dir, 
						      const sistrip::Granularity& gran ) {
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { delete generator_; generator_ = 0; }
  generator_ = SummaryGenerator::instance( view );
  
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<VpspScanAnalysis>::extract( const std::map<uint32_t,VpspScanAnalysis>& data  ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " No data to histogram!";
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
  std::map<uint32_t,VpspScanAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( mon_ == sistrip::VPSP_SCAN_BOTH_APVS ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.vpsp()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second.vpsp()[1] ); 
    } else if ( mon_ == sistrip::VPSP_SCAN_APV0 ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.vpsp()[0] ); 
    } else if ( mon_ == sistrip::VPSP_SCAN_APV1 ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.vpsp()[1] ); 
    } else { 
      edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	   << " Unexpected SummaryHisto value:"
	   << SiStripEnumsAndStrings::monitorable( mon_ ) 
	  ;
      continue;
    }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<VpspScanAnalysis>::fill( TH1& summary_histo ) {

  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
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
    generator_->summaryHisto( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SUM ) {
    generator_->summary1D( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SCATTER ) {
    generator_->summary2D( summary_histo );
  } else if ( pres_ == sistrip::PROFILE_1D ) {
    generator_->summaryProf( summary_histo );
  } else { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " Unexpected SummaryType value:"
	 << SiStripEnumsAndStrings::presentation( pres_ ) 
	;
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::VPSP_SCAN_BOTH_APVS ) {
  } else if ( mon_ == sistrip::VPSP_SCAN_APV0 ) { 
  } else if ( mon_ == sistrip::VPSP_SCAN_APV1 ) {
  } else { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
	 << " Unexpected SummaryHisto value:"
	 << SiStripEnumsAndStrings::monitorable( mon_ ) 
	;
  } 
  generator_->format( sistrip::VPSP_SCAN, mon_, pres_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<VpspScanAnalysis>;

