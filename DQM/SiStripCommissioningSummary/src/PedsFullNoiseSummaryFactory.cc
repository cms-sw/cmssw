#include "DQM/SiStripCommissioningSummary/interface/PedsFullNoiseSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void PedsFullNoiseSummaryFactory::extract( Iterator iter ) {
  
  PedsFullNoiseAnalysis* anal = dynamic_cast<PedsFullNoiseAnalysis*>( iter->second );
  if ( !anal ) { return; }
    
  std::vector< float > 				temp(128, 1. * sistrip::invalid_ );
  std::vector< std::vector<float> > value( 2, temp );
  std::vector< std::vector<float> > peds( 2, temp );
  std::vector< std::vector<float> > noise( 2, temp );
  std::vector< std::vector<float> > ks( 2, temp );
  std::vector< std::vector<float> > noiseG( 2, temp );
  std::vector< std::vector<float> > bin84( 2, temp );
  std::vector< std::vector<float> > chi2( 2, temp );
  std::vector< std::vector<float> > signif( 2, temp );
  std::vector< std::vector<float> > rms( 2, temp );
  peds[0] 	= anal->peds()[0];
  peds[1] 	= anal->peds()[1];
  noise[0] 	= anal->noise()[0];
  noise[1] 	= anal->noise()[1];
  ks[0] 	= anal->ksProb()[0]; // dummy values //replaced with ksProb now, wing
  ks[1] 	= anal->ksProb()[1]; // dummy values
  noiseG[0] 	= anal->noiseGaus()[0];
  noiseG[1] 	= anal->noiseGaus()[1];
  bin84[0] 	= anal->noiseBin84()[0];
  bin84[1] 	= anal->noiseBin84()[1];
  rms[0] 	= anal->noiseRMS()[0];
  rms[1] 	= anal->noiseRMS()[1];
  chi2[0]     = anal->chi2Prob()[0];
  chi2[1]     = anal->chi2Prob()[1];
  signif[0]     = anal->noiseSignif()[0];
  signif[1]     = anal->noiseSignif()[1];
  bool all_strips = false;
  if ( mon_ == sistrip::PEDESTALS_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = peds[0].size();
    if ( peds[0].size() < peds[1].size() ) { bins = peds[1].size(); }
    for ( uint16_t iped = 0; iped < bins; iped++ ) {
      value[0][iped] = peds[0][iped]; 
      value[1][iped] = peds[1][iped];  
    }
  } else if ( mon_ == sistrip::PEDESTALS_MEAN ) {
    value[0][0] = anal->pedsMean()[0];
    value[1][0] = anal->pedsMean()[1];
  } else if ( mon_ == sistrip::PEDESTALS_SPREAD ) { 
    value[0][0] = anal->pedsSpread()[0];
    value[1][0] = anal->pedsSpread()[1]; 
  } else if ( mon_ == sistrip::PEDESTALS_MAX ) { 
    value[0][0] = anal->pedsMax()[0]; 
    value[1][0] = anal->pedsMax()[1];
  } else if ( mon_ == sistrip::PEDESTALS_MIN ) { 
    value[0][0] = anal->pedsMin()[0]; 
    value[1][0] = anal->pedsMin()[1]; 
  } else if ( mon_ == sistrip::NOISE_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = noise[0].size();
    if ( noise[0].size() < noise[1].size() ) { bins = noise[1].size(); }
    for ( uint16_t inoise = 0; inoise < bins; inoise++ ) {
      value[0][inoise] = noise[0][inoise]; 
      value[1][inoise] = noise[1][inoise]; 
    }
  } else if ( mon_ == sistrip::NOISE_KS_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = ks[0].size();
    if ( ks[0].size() < ks[1].size() ) { bins = ks[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = ks[0][iks];
      value[1][iks] = ks[1][iks];
    }
  } else if ( mon_ == sistrip::NOISE_CHI2_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = chi2[0].size();
    if ( ks[0].size() < chi2[1].size() ) { bins = chi2[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = chi2[0][iks];
      value[1][iks] = chi2[1][iks];
    }
  } else if ( mon_ == sistrip::NOISE_GAUS_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = noiseG[0].size();
	if ( noiseG[0].size() < noiseG[1].size() ) { bins = noiseG[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = noiseG[0][iks];
      value[1][iks] = noiseG[1][iks];     
  	}
  } else if ( mon_ == sistrip::NOISE_BIN_84_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = bin84[0].size();
	if ( bin84[0].size() < bin84[1].size() ) { bins = bin84[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = bin84[0][iks];
      value[1][iks] = bin84[1][iks];     
  	}
  }	else if ( mon_ == sistrip::NOISE_RMS_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = rms[0].size();
	if ( rms[0].size() < rms[1].size() ) { bins = rms[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = rms[0][iks];
      value[1][iks] = rms[1][iks];     
  	}
  } else if ( mon_ == sistrip::NOISE_SIGNIF_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = signif[0].size();
	if ( signif[0].size() < signif[1].size() ) { bins = signif[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = signif[0][iks];
      value[1][iks] = signif[1][iks];     
  	}
  } else if ( mon_ == sistrip::NOISE_MEAN ) {
    value[0][0] = anal->noiseMean()[0];
    value[1][0] = anal->noiseMean()[1];
  } else if ( mon_ == sistrip::NOISE_SPREAD ) { 
    value[0][0] = anal->noiseSpread()[0]; 
    value[1][0] = anal->noiseSpread()[1];
  } else if ( mon_ == sistrip::NOISE_MAX ) { 
    value[0][0] = anal->noiseMax()[0]; 
    value[1][0] = anal->noiseMax()[1]; 
  } else if ( mon_ == sistrip::NOISE_MIN ) { 
    value[0][0] = anal->noiseMin()[0]; 
    value[1][0] = anal->noiseMin()[1]; 
  } else if ( mon_ == sistrip::NUM_OF_DEAD ) { 
    value[0][0] = 1. * anal->dead()[0].size(); 
    value[1][0] = 1. * anal->dead()[1].size();
  } else if ( mon_ == sistrip::NUM_OF_NOISY ) { 
    value[0][0] = 1. * anal->noisy()[0].size(); 
    value[1][0] = 1. * anal->noisy()[1].size();
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected monitorable: "
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
    return; 
  }

  if ( !all_strips ) {

    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value[0][0] );
    
    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value[1][0] );

  } 
  else {

    for ( uint16_t istr = 0; istr < value[0].size(); istr++ ) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						   SummaryPlotFactoryBase::gran_, 
						   iter->first, 
						   value[0][istr] );
    }
    
    for ( uint16_t istr = 0; istr < value[1].size(); istr++ ) {
      SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						   SummaryPlotFactoryBase::gran_, 
						   iter->first, 
						   value[1][istr] );
    }
  }
}

// -----------------------------------------------------------------------------
//
void PedsFullNoiseSummaryFactory::format() {
  
  if ( mon_ == sistrip::PEDESTALS_ALL_STRIPS ) {
    generator_->axisLabel( "Pedestal value [adc]" );
  } else if ( mon_ == sistrip::PEDESTALS_MEAN ) {
  } else if ( mon_ == sistrip::PEDESTALS_SPREAD ) { 
  } else if ( mon_ == sistrip::PEDESTALS_MAX ) { 
  } else if ( mon_ == sistrip::PEDESTALS_MIN ) { 
  } else if ( mon_ == sistrip::NOISE_ALL_STRIPS ) {
    generator_->axisLabel( "Noise [adc]" );
  } else if ( mon_ == sistrip::NOISE_MEAN ) {
  } else if ( mon_ == sistrip::NOISE_SPREAD ) { 
  } else if ( mon_ == sistrip::NOISE_MAX ) { 
  } else if ( mon_ == sistrip::NOISE_MIN ) { 
  } else if ( mon_ == sistrip::NUM_OF_DEAD ) { 
  } else if ( mon_ == sistrip::NUM_OF_NOISY ) { 
  } else if ( mon_ == sistrip::NOISE_KS_ALL_STRIPS ) { 
    generator_->axisLabel( "KS Prob." );
  } else if ( mon_ == sistrip::NOISE_GAUS_ALL_STRIPS ) { 
    generator_->axisLabel( "Noise Gaus." );
  } else if ( mon_ == sistrip::NOISE_BIN_84_ALL_STRIPS ) { 
    generator_->axisLabel( "Noise Bin 84." );
  }	else if ( mon_ == sistrip::NOISE_RMS_ALL_STRIPS ) { 
    generator_->axisLabel( "Noise RMS." );
  } else if ( mon_ == sistrip::NOISE_CHI2_ALL_STRIPS ) { 
    generator_->axisLabel( "Chi2 Prob." );
  }  else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 

}
