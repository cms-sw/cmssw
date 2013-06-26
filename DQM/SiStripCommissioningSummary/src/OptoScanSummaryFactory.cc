#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void OptoScanSummaryFactory::extract( Iterator iter ) {
  
  OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( iter->second );
  if ( !anal ) { return; }

  uint16_t igain = anal->gain();
  if ( igain > sistrip::valid_ ) { return; }
  
  float value = 1. * sistrip::invalid_;
  float error = 1. * sistrip::invalid_;
  
  if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
    value = igain; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) { 
    value = anal->bias()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
    value = anal->measGain()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
    value = anal->zeroLight()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LINK_NOISE ) { 
    value = anal->linkNoise()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) { 
    value = anal->liftOff()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LASER_THRESHOLD ) { 
    value = anal->threshold()[igain]; 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_TICK_HEIGHT ) { 
    value = anal->tickHeight()[igain]; 
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
					       value,
					       error );
  
}

// -----------------------------------------------------------------------------
//
void OptoScanSummaryFactory::format() {
  
  if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) {
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LINK_NOISE ) {
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) {
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LASER_THRESHOLD ) {
  } else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_TICK_HEIGHT ) {
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 

}
