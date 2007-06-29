#include "DQM/SiStripCommissioningSummary/interface/FedCablingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<FedCablingAnalysis>::SummaryHistogramFactory() :
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
SummaryHistogramFactory<FedCablingAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedCablingAnalysis>::init( const sistrip::Monitorable& mon, 
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
uint32_t SummaryHistogramFactory<FedCablingAnalysis>::extract( const std::map<uint32_t,FedCablingAnalysis>& data  ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " No data to histogram!";
    return 0; 
  } 
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return 0;  
  }
  
  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  std::map<uint32_t,FedCablingAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( mon_ == sistrip::FED_CABLING_FED_ID ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.fedId() ); 
    } else if ( mon_ == sistrip::FED_CABLING_FED_CH ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.fedCh() ); 
    } else if ( mon_ == sistrip::FED_CABLING_ADC_LEVEL ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.adcLevel() ); 
    } else { 
      edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	   << " Unexpected SummaryHisto value: "
	   << SiStripEnumsAndStrings::monitorable( mon_ ) 
	  ;
      continue;
    }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedCablingAnalysis>::fill( TH1& summary_histo ) {

  // Check if instance of generator class exists
  if ( !generator_ ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if std::map is filled
  if ( !generator_->size() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " No data in the monitorables std::map!";
    return; 
  } 

  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::SUMMARY_HISTO ) {
    generator_->summaryHisto( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_1D ) {
    generator_->summary1D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_2D ) {
    generator_->summary2D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_PROF ) {
    generator_->summaryProf( summary_histo );
  } else { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " Unexpected SummaryType value:"
	 << SiStripEnumsAndStrings::presentation( pres_ ) 
	;
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::FED_CABLING_FED_ID ) {
    generator_->axisLabel( "FED id" );
  } else if ( mon_ == sistrip::FED_CABLING_FED_CH ) { 
    generator_->axisLabel( "FED channel" );
  } else if ( mon_ == sistrip::FED_CABLING_ADC_LEVEL ) { 
    generator_->axisLabel( "ADC level" );
  } else { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory<FedCablingAnalysis>::" << __func__ << "]" 
	 << " Unexpected SummaryHisto value:"
	 << SiStripEnumsAndStrings::monitorable( mon_ ) 
	;
  } 
  generator_->format( sistrip::FED_CABLING, mon_, pres_, view_, level_, gran_, summary_histo );
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<FedCablingAnalysis>;

