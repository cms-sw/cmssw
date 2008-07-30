#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<CalibrationAnalysis*>::init(const sistrip::Monitorable& mon, 
                                                        const sistrip::Presentation& pres,
                                                        const sistrip::View& view, 
                                                        const std::string& level, 
                                                        const sistrip::Granularity& gran,
                                                        const std::map<uint32_t,CalibrationAnalysis*>& data ) {

  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );

  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Check if data are present
  if ( data.empty() ) { 
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]" 
         << " No data to histogram!";
    return 0; 
  } 
  
  // Extract monitorable
  std::map<uint32_t,CalibrationAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); ++iter ) {
    if ( !iter->second ) { continue; }
    if ( mon_ == sistrip::CALIBRATION_AMPLITUDE ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->amplitudeMean()[0], iter->second->amplitudeSpread()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->amplitudeMean()[1], iter->second->amplitudeSpread()[1] ); 
    } else if ( mon_ == sistrip::CALIBRATION_TAIL ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->tailMean()[0],  iter->second->tailSpread()[0]); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->tailMean()[1],  iter->second->tailSpread()[1]); 
    } else if ( mon_ == sistrip::CALIBRATION_RISETIME ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->riseTimeMean()[0], iter->second->riseTimeSpread()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->riseTimeMean()[1], iter->second->riseTimeSpread()[1] ); 
    } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->timeConstantMean()[0], iter->second->timeConstantSpread()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->timeConstantMean()[1], iter->second->timeConstantSpread()[1] ); 
    } else if ( mon_ == sistrip::CALIBRATION_SMEARING ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->smearingMean()[0], iter->second->smearingSpread()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->smearingMean()[1], iter->second->smearingSpread()[1] ); 
    } else if ( mon_ == sistrip::CALIBRATION_CHI2 ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second->chi2Mean()[0], iter->second->chi2Spread()[0] ); 
      generator_->fillMap( level_, gran_, iter->first, iter->second->chi2Mean()[1], iter->second->chi2Spread()[1] ); 
    } else { 
      edm::LogWarning(mlSummaryPlots_)
          << "[SummaryPlotFactory::" << __func__ << "]"
          << " Unexpected monitorable: "
          << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
      continue; 
    }
    //TODO: fill also the information per strip directly
    //TODO: add more plots with min/max/spread
  }
  return SummaryPlotFactoryBase::generator_->nBins();
}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<CalibrationAnalysis*>::fill( TH1& summary_histo ) {

  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }

  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::HISTO_1D ) {
    generator_->histo1D( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SUM ) {
    generator_->histo2DSum( summary_histo );
  } else if ( pres_ == sistrip::HISTO_2D_SCATTER ) {
    generator_->histo2DScatter( summary_histo );
  } else if ( pres_ == sistrip::PROFILE_1D ) {
    generator_->profile1D( summary_histo );
  } else { 
    edm::LogWarning(mlSummaryPlots_) << " Unexpected SummaryType value:"
         << SiStripEnumsAndStrings::presentation( pres_ );
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::CALIBRATION_AMPLITUDE ) {
    generator_->axisLabel( "Amplitude (ADC)" );
  } else if ( mon_ == sistrip::CALIBRATION_TAIL ) { 
    generator_->axisLabel( "Tail (%)" );
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME ) { 
    generator_->axisLabel( "Rise time (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT ) { 
    generator_->axisLabel( "Time constant (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING ) { 
    generator_->axisLabel( "Smearing (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_CHI2 ) { 
    generator_->axisLabel( "Chi2" );
  } else { 
    edm::LogWarning(mlSummaryPlots_) 
         << "[SummaryPlotFactory::" << __func__ << "]"
         <<  " Unexpected SummaryHisto value:"
         << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ ) ;
  } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<CalibrationAnalysis*>;

