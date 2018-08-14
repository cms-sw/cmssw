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
  
  std::vector< float > temp(128, 1. * sistrip::invalid_ );
  std::vector< uint16_t > temp2(128, sistrip::invalid_);

  std::vector< std::vector<float> > value (2, temp);
  std::vector< std::vector<float> > peds (2, temp);
  std::vector< std::vector<float> > noise(2, temp);
  std::vector< std::vector<float> > adProbab(2, temp);
  std::vector< std::vector<float> > ksProbab(2, temp);
  std::vector< std::vector<float> > jbProbab(2, temp);
  std::vector< std::vector<float> > chi2Probab(2, temp);
  std::vector< std::vector<float> > residualRMS(2, temp);
  std::vector< std::vector<float> > residualGaus(2, temp);
  std::vector< std::vector<float> > noiseSignificance(2, temp);
  std::vector< std::vector<float> > residualMean(2, temp);
  std::vector< std::vector<float> > residualSkewness(2, temp);
  std::vector< std::vector<float> > residualKurtosis(2, temp);
  std::vector< std::vector<float> > residualIntegralNsigma(2, temp);
  std::vector< std::vector<float> > residualIntegral(2, temp);
  std::vector< std::vector<uint16_t> > badStripBit(2, temp2);
  std::vector< std::vector<uint16_t> > deadStripBit(2, temp2);

  // pedestal values
  peds[0] 	= anal->peds()[0];
  peds[1] 	= anal->peds()[1];
  // noise values
  noise[0] 	= anal->noise()[0];
  noise[1] 	= anal->noise()[1];
  // AD probab
  adProbab[0] 	= anal->adProbab()[0]; 
  adProbab[1] 	= anal->adProbab()[1]; 
  // KS probab
  ksProbab[0] 	= anal->ksProbab()[0]; 
  ksProbab[1] 	= anal->ksProbab()[1]; 
  // JB probab
  jbProbab[0] 	= anal->jbProbab()[0]; 
  jbProbab[1] 	= anal->jbProbab()[1]; 
  // CHI2 probab
  chi2Probab[0] = anal->chi2Probab()[0]; 
  chi2Probab[1] = anal->chi2Probab()[1]; 
  // noise RMS
  chi2Probab[0] = anal->chi2Probab()[0]; 
  chi2Probab[1] = anal->chi2Probab()[1]; 
  // residual RMS
  residualRMS[0] = anal->residualRMS()[0]; 
  residualRMS[1] = anal->residualRMS()[1]; 
  // residual Sigma
  residualGaus[0] = anal->residualSigmaGaus()[0]; 
  residualGaus[1] = anal->residualSigmaGaus()[1]; 
  // noise Significance
  noiseSignificance[0] = anal->noiseSignificance()[0]; 
  noiseSignificance[1] = anal->noiseSignificance()[1]; 
  // residual mean
  residualMean[0] = anal->residualMean()[0]; 
  residualMean[1] = anal->residualMean()[1]; 
  // noise Skweness
  residualSkewness[0] = anal->residualSkewness()[0]; 
  residualSkewness[1] = anal->residualSkewness()[1]; 
  // noise Kurtosis
  residualKurtosis[0] = anal->residualKurtosis()[0]; 
  residualKurtosis[1] = anal->residualKurtosis()[1]; 
  // noise integral N sigma
  residualIntegralNsigma[0] = anal->residualIntegralNsigma()[0]; 
  residualIntegralNsigma[1] = anal->residualIntegralNsigma()[1]; 
  // noise integral N sigma
  residualIntegral[0] = anal->residualIntegral()[0]; 
  residualIntegral[1] = anal->residualIntegral()[1]; 
  // bit to indicate if a strip is flagged as bad or not
  residualIntegral[0] = anal->residualIntegral()[0]; 
  residualIntegral[1] = anal->residualIntegral()[1]; 
  // bit to indicate if a strip is bad (1) or not (0)
  badStripBit[0] = anal->badStripBit()[0];
  badStripBit[1] = anal->badStripBit()[1];
  // bit to indicate if a strip is dead (1) or not (0)
  deadStripBit[0] = anal->deadStripBit()[0];
  deadStripBit[1] = anal->deadStripBit()[1];

  bool all_strips = false;
  // Monitor pedestals value for each strip
  if (mon_ == sistrip::PEDESTALS_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = peds[0].size();
    if (peds[0].size() < peds[1].size() ) { bins = peds[1].size(); }
    for( uint16_t iped = 0; iped < bins; iped++ ) {
      value[0][iped] = peds[0][iped]; 
      value[1][iped] = peds[1][iped];  
    }
  } 
  // Monitor noise value for each strip
  else if ( mon_ == sistrip::NOISE_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = noise[0].size();
    if ( noise[0].size() < noise[1].size() ) { bins = noise[1].size(); }
    for ( uint16_t inoise = 0; inoise < bins; inoise++ ) {
      value[0][inoise] = noise[0][inoise]; 
      value[1][inoise] = noise[1][inoise]; 
    }
  } 
  // Monitor pedestals aD probability for each strip
  else if ( mon_ == sistrip::AD_PROBAB_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = adProbab[0].size();
    if ( adProbab[0].size() < adProbab[1].size() ) { bins = adProbab[1].size(); }
    for ( uint16_t iad = 0; iad < bins; iad++ ) {
      value[0][iad] = adProbab[0][iad];
      value[1][iad] = adProbab[1][iad];
    }
  } 
  // Monitor pedestals KS probability for each strip
  else if ( mon_ == sistrip::KS_PROBAB_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = ksProbab[0].size();
    if ( ksProbab[0].size() < ksProbab[1].size() ) { bins = ksProbab[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = ksProbab[0][iks];
      value[1][iks] = ksProbab[1][iks];
    }
  } 
  // Monitor pedestals JB probability for each strip
  else if ( mon_ == sistrip::JB_PROBAB_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = jbProbab[0].size();
    if ( jbProbab[0].size() < jbProbab[1].size() ) { bins = jbProbab[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = jbProbab[0][iks];
      value[1][iks] = jbProbab[1][iks];
    }
  } 
  // Monitor pedestals Chi2 probability for each strip
  else if ( mon_ == sistrip::CHI2_PROBAB_ALL_STRIPS ) {
    all_strips = true;
    uint16_t bins = chi2Probab[0].size();
    if ( chi2Probab[0].size() < chi2Probab[1].size() ) { bins = chi2Probab[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = chi2Probab[0][iks];
      value[1][iks] = chi2Probab[1][iks];
    }
  }
  // Monitor pedestals RMS residual for each strip 
  else if ( mon_ == sistrip::RESIDUAL_RMS_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualRMS[0].size();
	if ( residualRMS[0].size() < residualRMS[1].size() ) { bins = residualRMS[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualRMS[0][iks];
      value[1][iks] = residualRMS[1][iks];     
    }
  } 
  // Monitor pedestals sigma from gaussian firt for each strip 
  else if ( mon_ == sistrip::RESIDUAL_GAUS_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualGaus[0].size();
    if ( residualGaus[0].size() < residualGaus[1].size() ) { bins = residualGaus[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualGaus[0][iks];
      value[1][iks] = residualGaus[1][iks];     
    }
  }   
  // Monitor pedestals noise significance for each strip
  else if ( mon_ == sistrip::NOISE_SIGNIFICANCE_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = noiseSignificance[0].size();
    if (noiseSignificance[0].size() < noiseSignificance[1].size() ) { bins = noiseSignificance[1].size(); }
    for (uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = noiseSignificance[0][iks];
      value[1][iks] = noiseSignificance[1][iks];     
    }
  }  
  // Monitor mean residual for each strip
  else if ( mon_ == sistrip::RESIDUAL_MEAN_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualMean[0].size();
    if ( residualMean[0].size() < residualMean[1].size() ) { bins = residualMean[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualMean[0][iks];
      value[1][iks] = residualMean[1][iks];     
    }
  } 
  // Monitor skweness for each strip
  else if ( mon_ == sistrip::RESIDUAL_SKEWNESS_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualSkewness[0].size();
    if ( residualSkewness[0].size() < residualSkewness[1].size() ) { bins = residualSkewness[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualSkewness[0][iks];
      value[1][iks] = residualSkewness[1][iks];     
    }
  }   
  // Monitor Kurtosis for each strip
  else if ( mon_ == sistrip::RESIDUAL_KURTOSIS_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualKurtosis[0].size();
    if ( residualKurtosis[0].size() < residualKurtosis[1].size() ) { bins = residualKurtosis[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualKurtosis[0][iks];
      value[1][iks] = residualKurtosis[1][iks];     
    }
  } 
  // Monitor Integral above N sigma for each strip
  else if ( mon_ == sistrip::RESIDUAL_INTEGRALNSIGMA_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualIntegralNsigma[0].size();
    if ( residualIntegralNsigma[0].size() < residualIntegralNsigma[1].size() ) { bins = residualIntegralNsigma[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualIntegralNsigma[0][iks];
      value[1][iks] = residualIntegralNsigma[1][iks];     
    }
  } 
  // Monitor integral for each strip
  else if ( mon_ == sistrip::RESIDUAL_INTEGRAL_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = residualIntegral[0].size();
	if ( residualIntegral[0].size() < residualIntegral[1].size() ) { bins = residualIntegral[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = residualIntegral[0][iks];
      value[1][iks] = residualIntegral[1][iks];     
    }
  } 

  // Monitor BadStrip bit
  else if ( mon_ == sistrip::BAD_STRIP_BIT_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = badStripBit[0].size();
    if ( badStripBit[0].size() < badStripBit[1].size() ) { bins = badStripBit[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = 1.*badStripBit[0][iks];
      value[1][iks] = 1.*badStripBit[1][iks];     
    }
  } 
  // Dead strip bit
  else if ( mon_ == sistrip::DEAD_STRIP_BIT_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = deadStripBit[0].size();
	if ( deadStripBit[0].size() < deadStripBit[1].size() ) { bins = deadStripBit[1].size(); }
    for ( uint16_t iks = 0; iks < bins; iks++ ) {
      value[0][iks] = 1.*deadStripBit[0][iks];
      value[1][iks] = 1.*deadStripBit[1][iks];     
    }
  } 
  
  // Per APV information: pedsMean
  else if ( mon_ == sistrip::PEDESTALS_MEAN ) {
    value[0][0] = anal->pedsMean()[0];
    value[1][0] = anal->pedsMean()[1];
  } 

  // Per APV information: pedsSpread
  else if ( mon_ == sistrip::PEDESTALS_SPREAD ) { 
    value[0][0] = anal->pedsSpread()[0];
    value[1][0] = anal->pedsSpread()[1]; 
  } 

  // Per APV information: pedsMax
  else if ( mon_ == sistrip::PEDESTALS_MAX ) { 
    value[0][0] = anal->pedsMax()[0]; 
    value[1][0] = anal->pedsMax()[1];
  } 

  // Per APV information: pedsMin
  else if ( mon_ == sistrip::PEDESTALS_MIN ) { 
    value[0][0] = anal->pedsMin()[0]; 
    value[1][0] = anal->pedsMin()[1]; 
  } 

  // Per APV information: noiseMean
  else if ( mon_ == sistrip::NOISE_MEAN ) {
    value[0][0] = anal->noiseMean()[0];
    value[1][0] = anal->noiseMean()[1];
  } 
  // Per APV information: noiseSpread
  else if ( mon_ == sistrip::NOISE_SPREAD ) { 
    value[0][0] = anal->noiseSpread()[0]; 
    value[1][0] = anal->noiseSpread()[1];
  } 
  // Per APV information: noiseMax
  else if ( mon_ == sistrip::NOISE_MAX ) { 
    value[0][0] = anal->noiseMax()[0]; 
    value[1][0] = anal->noiseMax()[1]; 
  } 
  // Per APV information: noiseMin
  else if ( mon_ == sistrip::NOISE_MIN ) { 
    value[0][0] = anal->noiseMin()[0]; 
    value[1][0] = anal->noiseMin()[1]; 
  }
  
  // BAD channels per APV  
  else if ( mon_ == sistrip::NUM_OF_DEAD ) { 
    value[0][0] = 1. * anal->deadStrip()[0].size(); 
    value[1][0] = 1. * anal->deadStrip()[1].size();
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD ) { 
    value[0][0] = 1. * anal->badStrip()[0].size(); 
    value[1][0] = 1. * anal->badStrip()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_SHIFTED) { 
    value[0][0] = 1. * anal->shiftedStrip()[0].size(); 
    value[1][0] = 1. * anal->shiftedStrip()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_LOW_NOISE) { 
    value[0][0] = 1. * anal->lowNoiseStrip()[0].size(); 
    value[1][0] = 1. * anal->lowNoiseStrip()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_LARGE_NOISE) { 
    value[0][0] = 1. * anal->largeNoiseStrip()[0].size(); 
    value[1][0] = 1. * anal->largeNoiseStrip()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_LARGE_SIGNIF) { 
    value[0][0] = 1. * anal->largeNoiseSignificance()[0].size(); 
    value[1][0] = 1. * anal->largeNoiseSignificance()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_FIT_STATUS) { 
    value[0][0] = 1. * anal->badFitStatus()[0].size(); 
    value[1][0] = 1. * anal->badFitStatus()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_AD_PROBAB) { 
    value[0][0] = 1. * anal->badADProbab()[0].size(); 
    value[1][0] = 1. * anal->badADProbab()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_KS_PROBAB) { 
    value[0][0] = 1. * anal->badKSProbab()[0].size(); 
    value[1][0] = 1. * anal->badKSProbab()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_JB_PROBAB) { 
    value[0][0] = 1. * anal->badJBProbab()[0].size(); 
    value[1][0] = 1. * anal->badJBProbab()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_CHI2_PROBAB) { 
    value[0][0] = 1. * anal->badChi2Probab()[0].size(); 
    value[1][0] = 1. * anal->badChi2Probab()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_TAIL) { 
    value[0][0] = 1. * anal->badTailStrip()[0].size(); 
    value[1][0] = 1. * anal->badTailStrip()[1].size();    
  } 
  else if ( mon_ == sistrip::NUM_OF_BAD_DOUBLE_PEAK) { 
    value[0][0] = 1. * anal->badDoublePeakStrip()[0].size(); 
    value[1][0] = 1. * anal->badDoublePeakStrip()[1].size();    
  }   
  else { 
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
  }
  else if ( mon_ == sistrip::PEDESTALS_MEAN ) {
  } 
  else if ( mon_ == sistrip::PEDESTALS_SPREAD ) { 
  } 
  else if ( mon_ == sistrip::PEDESTALS_MAX ) { 
  } 
  else if ( mon_ == sistrip::PEDESTALS_MIN ) { 
  } 
  else if ( mon_ == sistrip::NOISE_ALL_STRIPS ) {
    generator_->axisLabel( "Noise [adc]" );
  } 
  else if ( mon_ == sistrip::NOISE_MEAN ) {
  } 
  else if ( mon_ == sistrip::NOISE_SPREAD ) { 
  } 
  else if ( mon_ == sistrip::NOISE_MAX ) { 
  } 
  else if ( mon_ == sistrip::NOISE_MIN ) { 
  }
  else if( mon_ == sistrip::AD_PROBAB_ALL_STRIPS) {
    generator_->axisLabel("Anderson-Darling p-value" );    
  }
  else if( mon_ == sistrip::KS_PROBAB_ALL_STRIPS) {
    generator_->axisLabel("Kolmogorov-Smirnov p-value" );    
  }
  else if( mon_ == sistrip::JB_PROBAB_ALL_STRIPS) {
    generator_->axisLabel("Jacque-Bera p-value" );    
  }
  else if( mon_ == sistrip::CHI2_PROBAB_ALL_STRIPS) {
    generator_->axisLabel("Chi2 p-value" );    
  }
  else if( mon_ == sistrip::RESIDUAL_RMS_ALL_STRIPS) {
    generator_->axisLabel("Residual RMS [adc]" );    
  }
  else if( mon_ == sistrip::RESIDUAL_GAUS_ALL_STRIPS) {
    generator_->axisLabel("Residual Gaus [adc]" );    
  }
  else if( mon_ == sistrip::NOISE_SIGNIFICANCE_ALL_STRIPS) {
    generator_->axisLabel("Noise Significance" );    
  }
  else if( mon_ == sistrip::RESIDUAL_MEAN_ALL_STRIPS) {
    generator_->axisLabel("Residual Mean [adc]" );    
  }
  else if( mon_ == sistrip::RESIDUAL_SKEWNESS_ALL_STRIPS) {
    generator_->axisLabel("Residual Skewness [adc]" );    
  }
  else if( mon_ == sistrip::RESIDUAL_KURTOSIS_ALL_STRIPS) {
    generator_->axisLabel("Residual Kurtosis [adc]" );    
  }
  else if( mon_ == sistrip::RESIDUAL_INTEGRALNSIGMA_ALL_STRIPS) {
    generator_->axisLabel("Residual Integral at N sigma" );    
  }
  else if( mon_ == sistrip::RESIDUAL_INTEGRAL_ALL_STRIPS) {
    generator_->axisLabel("Residual Integral" );    
  }
  else if( mon_ == sistrip::BAD_STRIP_BIT_ALL_STRIPS) {
    generator_->axisLabel("Bad Strip Bit" );    
  }
  else if( mon_ == sistrip::DEAD_STRIP_BIT_ALL_STRIPS) {
    generator_->axisLabel("Dead Strip Bit" );    
  }
  else if( mon_ == sistrip::NUM_OF_DEAD) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_SHIFTED) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_LOW_NOISE) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_LARGE_NOISE) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_LARGE_SIGNIF) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_FIT_STATUS) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_AD_PROBAB) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_KS_PROBAB) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_JB_PROBAB) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_CHI2_PROBAB) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_TAIL) {
  }
  else if( mon_ == sistrip::NUM_OF_BAD_DOUBLE_PEAK) {
  }

  else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 

}
