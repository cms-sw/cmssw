#include "DQM/SiStripCommissioningSummary/interface/FastFedCablingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<FastFedCablingAnalysis*>::init( const sistrip::Monitorable& mon, 
							    const sistrip::Presentation& pres,
							    const sistrip::View& view, 
							    const std::string& level, 
							    const sistrip::Granularity& gran,
							    const std::map<uint32_t,FastFedCablingAnalysis*>& data ) {
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Extract monitorable
  std::map<uint32_t,FastFedCablingAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( !iter->second ) { continue; }
    float value = 1. * sistrip::invalid_;
    float error = 1. * sistrip::invalid_;
    if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_HIGH_LEVEL ) { 
      value = iter->second->highLevel(); 
      error = iter->second->highRms(); 
    } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_LOW_LEVEL ) { 
      value = iter->second->lowLevel(); 
      error = iter->second->lowRms(); 
    } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_MAX ) { 
      value = iter->second->max(); 
    } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_MIN ) { 
      value = iter->second->min(); 
      //} else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_CONNS_PER_FED ) { 
      //if ( iter->second->isValid() ) { value = 1.; } 
    } else { 
      edm::LogWarning(mlSummaryPlots_)
	<< "[SummaryPlotFactory::" << __func__ << "]" 
	<< " Unexpected monitorable: "
	<< SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
      continue; 
    }
    
    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value,
						 error );

  }
  
  SummaryPlotFactoryBase::generator_->printMap();
  return SummaryPlotFactoryBase::generator_->nBins();
  
}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<FastFedCablingAnalysis*>::fill( TH1& summary_histo ) {
  
  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }
  
  // Histogram formatting
  if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_HIGH_LEVEL ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "\"High\" light level [ADC]" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_LOW_LEVEL ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "\"Low\" light level [ADC]" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_MAX ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "Maximum light level [ADC]" );
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::FAST_FED_CABLING_MIN ) {
    SummaryPlotFactoryBase::generator_->axisLabel( "Minumum light level [ADC]" );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<FastFedCablingAnalysis*>;

