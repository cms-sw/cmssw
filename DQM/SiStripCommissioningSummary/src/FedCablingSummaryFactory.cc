#include "DQM/SiStripCommissioningSummary/interface/FedCablingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<FedCablingAnalysis*>::init( const sistrip::Monitorable& mon, 
							const sistrip::Presentation& pres,
							const sistrip::View& view, 
							const std::string& level, 
							const sistrip::Granularity& gran,
							const std::map<uint32_t,FedCablingAnalysis*>& data ) {
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Check if generator class exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Extract monitorable
  std::map<uint32_t,FedCablingAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    float value = static_cast<float>(sistrip::invalid_);
    if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_FED_ID ) { value = iter->second->fedId(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_FED_CH ) { value = iter->second->fedCh(); }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_ADC_LEVEL ) { value = iter->second->adcLevel(); }
    else { 
      edm::LogWarning(mlSummaryPlots_)
	<< "[SummaryPlotFactory::" << __func__ << "]" 
	<< " Unexpected monitorable: "
	<< SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
      continue; 
    }
    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value );
  }
  
  return SummaryPlotFactoryBase::generator_->nBins();
  
}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<FedCablingAnalysis*>::fill( TH1& summary_histo ) {
  
  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }
  
  // Histogram formatting
  if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_FED_ID ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "FED id" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_FED_CH ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "FED channel" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FED_CABLING_ADC_LEVEL ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "Signal level [ADC]" );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<FedCablingAnalysis*>;

