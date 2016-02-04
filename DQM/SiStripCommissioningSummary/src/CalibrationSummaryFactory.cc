#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void CalibrationSummaryFactory::extract( Iterator iter ) {

  CalibrationAnalysis* anal = dynamic_cast<CalibrationAnalysis*>( iter->second );
  if ( !anal ) { return; }

  std::vector<float> temp(128, 1. * sistrip::invalid_ );
  std::vector< std::vector<float> > value( 2, temp );
  std::vector< std::vector<float> > error( 2, temp );
  std::vector< std::vector<float> > amplitude( 2, temp );
  std::vector< std::vector<float> > tail( 2, temp );
  std::vector< std::vector<float> > riseTime( 2, temp );
  std::vector< std::vector<float> > timeConstant( 2, temp );
  std::vector< std::vector<float> > smearing( 2, temp );
  std::vector< std::vector<float> > chi2( 2, temp );
  amplitude[0] = anal->amplitude()[0];
  amplitude[1] = anal->amplitude()[1];
  tail[0] = anal->tail()[0];
  tail[1] = anal->tail()[1];
  riseTime[0] = anal->riseTime()[0];
  riseTime[1] = anal->riseTime()[1];
  timeConstant[0] = anal->timeConstant()[0];
  timeConstant[1] = anal->timeConstant()[1];
  smearing[0] = anal->smearing()[0];
  smearing[1] = anal->smearing()[1];
  chi2[0] = anal->chi2()[0];
  chi2[1] = anal->chi2()[1];
  
  SiStripFecKey lldKey = SiStripFecKey(iter->first);
  uint32_t key1 = SiStripFecKey(lldKey.fecCrate(),
                                lldKey.fecSlot(),
                                lldKey.fecRing(),
                                lldKey.ccuAddr(),
                                lldKey.ccuChan(),
                                lldKey.lldChan(),
                                lldKey.i2cAddr(lldKey.lldChan(),true)).key();
  uint32_t key2 = SiStripFecKey(lldKey.fecCrate(),
                                lldKey.fecSlot(),
                                lldKey.fecRing(),
                                lldKey.ccuAddr(),
                                lldKey.ccuChan(),
                                lldKey.lldChan(),
                                lldKey.i2cAddr(lldKey.lldChan(),false)).key();
  
  bool all_strips = false;
  bool with_error = false;
  if ( mon_ == sistrip::CALIBRATION_AMPLITUDE_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = amplitude[amplitude[0].size() < amplitude[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = amplitude[0][i]/10.;
      value[1][i] = amplitude[1][i]/10.;
    }
  } else if ( mon_ == sistrip::CALIBRATION_AMPLITUDE) {
    with_error = true;
    value[0][0] = anal->amplitudeMean()[0]/10.;
    value[1][0] = anal->amplitudeMean()[1]/10.;
    error[0][0] = anal->amplitudeSpread()[0]/10.;
    error[1][0] = anal->amplitudeSpread()[1]/10.;
  } else if ( mon_ == sistrip::CALIBRATION_AMPLITUDE_MIN) {
    value[0][0] = anal->amplitudeMin()[0]/10.;
    value[1][0] = anal->amplitudeMin()[1]/10.;
  } else if ( mon_ == sistrip::CALIBRATION_AMPLITUDE_MAX) {
    value[0][0] = anal->amplitudeMax()[0]/10.;
    value[1][0] = anal->amplitudeMax()[1]/10.;
  } else if ( mon_ == sistrip::CALIBRATION_TAIL_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = tail[tail[0].size() < tail[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = tail[0][i];
      value[1][i] = tail[1][i];
    }
  } else if ( mon_ == sistrip::CALIBRATION_TAIL) {
    with_error = true;
    value[0][0] = anal->tailMean()[0];
    value[1][0] = anal->tailMean()[1];
    error[0][0] = anal->tailSpread()[0];
    error[1][0] = anal->tailSpread()[1];
  } else if ( mon_ == sistrip::CALIBRATION_TAIL_MIN) {
    value[0][0] = anal->tailMin()[0];
    value[1][0] = anal->tailMin()[1];
  } else if ( mon_ == sistrip::CALIBRATION_TAIL_MAX) {
    value[0][0] = anal->tailMax()[0];
    value[1][0] = anal->tailMax()[1];
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = riseTime[riseTime[0].size() < riseTime[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = riseTime[0][i];
      value[1][i] = riseTime[1][i];
    }
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME) {
    with_error = true;
    value[0][0] = anal->riseTimeMean()[0];
    value[1][0] = anal->riseTimeMean()[1];
    error[0][0] = anal->riseTimeSpread()[0];
    error[1][0] = anal->riseTimeSpread()[1];
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME_MIN) {
    value[0][0] = anal->riseTimeMin()[0];
    value[1][0] = anal->riseTimeMin()[1];
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME_MAX) {
    value[0][0] = anal->riseTimeMax()[0];
    value[1][0] = anal->riseTimeMax()[1];
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = timeConstant[timeConstant[0].size() < timeConstant[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = timeConstant[0][i];
      value[1][i] = timeConstant[1][i];
    }
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT) {
    with_error = true;
    value[0][0] = anal->timeConstantMean()[0];
    value[1][0] = anal->timeConstantMean()[1];
    error[0][0] = anal->timeConstantSpread()[0];
    error[1][0] = anal->timeConstantSpread()[1];
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT_MIN) {
    value[0][0] = anal->timeConstantMin()[0];
    value[1][0] = anal->timeConstantMin()[1];
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT_MAX) {
    value[0][0] = anal->timeConstantMax()[0];
    value[1][0] = anal->timeConstantMax()[1];
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = smearing[smearing[0].size() < smearing[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = smearing[0][i];
      value[1][i] = smearing[1][i];
    }
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING) {
    with_error = true;
    value[0][0] = anal->smearingMean()[0];
    value[1][0] = anal->smearingMean()[1];
    error[0][0] = anal->smearingSpread()[0];
    error[1][0] = anal->smearingSpread()[1];
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING_MIN) {
    value[0][0] = anal->smearingMin()[0];
    value[1][0] = anal->smearingMin()[1];
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING_MAX) {
    value[0][0] = anal->smearingMax()[0];
    value[1][0] = anal->smearingMax()[1];
  } else if ( mon_ == sistrip::CALIBRATION_CHI2_ALLSTRIPS) {
    all_strips = true;
    uint16_t bins = chi2[chi2[0].size() < chi2[1].size() ? 1 : 0].size();
    for ( uint16_t i = 0; i < bins; i++ ) {
      value[0][i] = chi2[0][i];
      value[1][i] = chi2[1][i];
    }
  } else if ( mon_ == sistrip::CALIBRATION_CHI2) {
    with_error = true;
    value[0][0] = anal->chi2Mean()[0]/100.;
    value[1][0] = anal->chi2Mean()[1]/100.;
    error[0][0] = anal->chi2Spread()[0]/100.;
    error[1][0] = anal->chi2Spread()[1]/100.;
  } else if ( mon_ == sistrip::CALIBRATION_CHI2_MIN) {
    value[0][0] = anal->chi2Min()[0]/100.;
    value[1][0] = anal->chi2Min()[1]/100.;
  } else if ( mon_ == sistrip::CALIBRATION_CHI2_MAX) {
    value[0][0] = anal->chi2Max()[0]/100.;
    value[1][0] = anal->chi2Max()[1]/100.;
  } else { 
    edm::LogWarning(mlSummaryPlots_)
        << "[SummaryPlotFactory::" << __func__ << "]"
        << " Unexpected monitorable: "
        << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
    return; 
  }

  if ( !all_strips ) {
    if( !with_error) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
  						   key1,
  						   value[0][0] );
  						 
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
  						   key2,
  						   value[1][0] );
    } else {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
  						   key1,
  						   value[0][0],
						   error[0][0]);
  						 
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
  						   key2,
  						   value[1][0],
						   error[1][0]);
    }
  } else {
    
    for ( uint16_t istr = 0; istr < value[0].size(); istr++ ) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
						   key1,
						   value[0][istr] );
    }
    
    for ( uint16_t istr = 0; istr < value[1].size(); istr++ ) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_,
                                                   SummaryPlotFactoryBase::gran_,
						   key2,
						   value[1][istr] );
    }
  }
}

//------------------------------------------------------------------------------
//
void CalibrationSummaryFactory::format() {

  // Histogram formatting
  if ( mon_ == sistrip::CALIBRATION_AMPLITUDE ) {
    generator_->axisLabel( "Amplitude (ADC*Nevt/10.)" );
  } else if ( mon_ == sistrip::CALIBRATION_TAIL ) { 
    generator_->axisLabel( "Tail (%)" );
  } else if ( mon_ == sistrip::CALIBRATION_RISETIME ) { 
    generator_->axisLabel( "Rise time (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_TIMECONSTANT ) { 
    generator_->axisLabel( "Time constant (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_SMEARING ) { 
    generator_->axisLabel( "Smearing (ns)" );
  } else if ( mon_ == sistrip::CALIBRATION_CHI2 ) { 
    generator_->axisLabel( "Chi2/100." );
  } else { 
    edm::LogWarning(mlSummaryPlots_) 
         << "[SummaryPlotFactory::" << __func__ << "]"
         <<  " Unexpected SummaryHisto value:"
         << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ ) ;
  } 
  
}

