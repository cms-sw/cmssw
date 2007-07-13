#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<VpspScanAnalysis*>::init( const sistrip::Monitorable& mon, 
						      const sistrip::Presentation& pres,
						      const sistrip::View& view, 
						      const std::string& level, 
						      const sistrip::Granularity& gran,
						      const std::map<uint32_t,VpspScanAnalysis*>& data ) {
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Extract monitorable
  std::map<uint32_t,VpspScanAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( !iter->second ) { continue; }
    std::vector<float> value( 2, 1. * sistrip::invalid_ );
    std::vector<float> error( 2, 1. * sistrip::invalid_ );
    bool two = true;
    if ( mon_ == sistrip::VPSP_SCAN_APV_SETTINGS ) {
      value[0] = 1. * iter->second->vpsp()[0]; 
      value[1] = 1. * iter->second->vpsp()[1]; 
    } else if ( mon_ == sistrip::VPSP_SCAN_APV0_SETTING ) {
      value[0] = 1. * iter->second->vpsp()[0]; 
      two = false;
    } else if ( mon_ == sistrip::VPSP_SCAN_APV1_SETTING ) {
      value[0] = 1. * iter->second->vpsp()[1]; 
      two = false;
    } else if ( mon_ == sistrip::VPSP_SCAN_ADC_LEVEL ) {
      value[0] = 1. * iter->second->adcLevel()[0]; 
      value[1] = 1. * iter->second->adcLevel()[1]; 
    } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_HIGH ) {
      value[0] = 1. * iter->second->topLevel()[0]; 
      value[1] = 1. * iter->second->topLevel()[1]; 
    } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_LOW ) {
      value[0] = 1. * iter->second->bottomLevel()[0]; 
      value[1] = 1. * iter->second->bottomLevel()[1]; 
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
						 value[0],
						 error[0] );

    if ( two ) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						   SummaryPlotFactoryBase::gran_, 
						   iter->first, 
						   value[1],
						   error[1] );
    }

  }
  
  return SummaryPlotFactoryBase::generator_->nBins();

}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<VpspScanAnalysis*>::fill( TH1& summary_histo ) {

  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }
  
  // Histogram formatting
  if ( mon_ == sistrip::VPSP_SCAN_APV_SETTINGS ) {
  } else if ( mon_ == sistrip::VPSP_SCAN_APV0_SETTING ) { 
  } else if ( mon_ == sistrip::VPSP_SCAN_APV1_SETTING ) {
  } else if ( mon_ == sistrip::VPSP_SCAN_ADC_LEVEL ) {
  } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_HIGH ) {
  } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_LOW ) {
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value: " 
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<VpspScanAnalysis*>;

