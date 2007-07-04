#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<OptoScanAnalysis*>::init( const sistrip::Monitorable& mon, 
						      const sistrip::Presentation& pres,
						      const sistrip::View& view, 
						      const std::string& level, 
						      const sistrip::Granularity& gran,
						      const std::map<uint32_t,OptoScanAnalysis*>& data ) {

  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );

  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }

  // Extract monitorable and fill map
  std::map<uint32_t,OptoScanAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( !iter->second ) { continue; }
    uint16_t igain = iter->second->gain();
    float value = 1. * sistrip::invalid_;
    //float error = 1. * sistrip::invalid_;
    if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { value = igain; }
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) { value = iter->second->bias()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_MEASURED_GAIN ) { value = iter->second->measGain()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { value = iter->second->zeroLight()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LINK_NOISE ) { value = iter->second->linkNoise()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) { value = iter->second->liftOff()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_LASER_THRESHOLD ) { value = iter->second->threshold()[igain]; } 
    else if ( SummaryPlotFactoryBase::mon_ == sistrip::OPTO_SCAN_TICK_HEIGHT ) { value = iter->second->tickHeight()[igain]; } 
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
void SummaryPlotFactory<OptoScanAnalysis*>::fill( TH1& summary_histo ) {

  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );

  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }

  // Histogram formatting
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

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<OptoScanAnalysis*>;

