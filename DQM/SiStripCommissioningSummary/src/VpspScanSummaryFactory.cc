#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/VpspScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void VpspScanSummaryFactory::extract( Iterator iter ) {
  
  VpspScanAnalysis* anal = dynamic_cast<VpspScanAnalysis*>( iter->second );
  if ( !anal ) { return; }
    
  std::vector<float> value( 2, 1. * sistrip::invalid_ );
  std::vector<float> error( 2, 1. * sistrip::invalid_ );

  bool two = true;
  if ( mon_ == sistrip::VPSP_SCAN_APV_SETTINGS ) {
    value[0] = 1. * anal->vpsp()[0]; 
    value[1] = 1. * anal->vpsp()[1]; 
  } else if ( mon_ == sistrip::VPSP_SCAN_APV0_SETTING ) {
    value[0] = 1. * anal->vpsp()[0]; 
    two = false;
  } else if ( mon_ == sistrip::VPSP_SCAN_APV1_SETTING ) {
    value[0] = 1. * anal->vpsp()[1]; 
    two = false;
  } else if ( mon_ == sistrip::VPSP_SCAN_ADC_LEVEL ) {
    value[0] = 1. * anal->adcLevel()[0]; 
    value[1] = 1. * anal->adcLevel()[1]; 
  } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_HIGH ) {
    value[0] = 1. * anal->topLevel()[0]; 
    value[1] = 1. * anal->topLevel()[1]; 
  } else if ( mon_ == sistrip::VPSP_SCAN_DIGITAL_LOW ) {
    value[0] = 1. * anal->bottomLevel()[0]; 
    value[1] = 1. * anal->bottomLevel()[1]; 
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected monitorable: "
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
    return; 
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

// -----------------------------------------------------------------------------
//
void VpspScanSummaryFactory::format() {
  
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
