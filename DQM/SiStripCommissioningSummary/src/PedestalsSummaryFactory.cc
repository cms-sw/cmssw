#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<PedestalsAnalysis*>::init( const sistrip::Monitorable& mon, 
						       const sistrip::Presentation& pres,
						       const sistrip::View& view, 
						       const std::string& level, 
						       const sistrip::Granularity& gran,
						       const std::map<uint32_t,PedestalsAnalysis*>& data ) {
  
  // Some initialisation
  SummaryPlotFactoryBase::init( mon, pres, view, level, gran );
  
  // Check if generator object exists
  if ( !SummaryPlotFactoryBase::generator_ ) { return 0; }
  
  // Extract monitorable
  std::map<uint32_t,PedestalsAnalysis*>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( !iter->second ) { continue; }
    std::vector<float> value( 2, 1. * sistrip::invalid_ );
    std::vector<float> error( 2, 1. * sistrip::invalid_ );
    std::vector<float> temp(0);
    std::vector< std::vector<float> > peds(2,temp);
    std::vector< std::vector<float> > noise(2,temp);
    peds[0] = iter->second->peds()[0];
    peds[1] = iter->second->peds()[1];
    noise[0] = iter->second->noise()[0];
    noise[1] = iter->second->noise()[1];
    if ( mon_ == sistrip::PEDESTALS_ALL_STRIPS ) {
      uint16_t bins = peds[0].size();
      if ( peds[0].size() < peds[1].size() ) { bins = peds[1].size(); }
      for ( uint16_t iped = 0; iped < bins; iped++ ) {
	value[0] = peds[0][iped]; 
	value[1] = peds[1][iped];  
      }
    } else if ( mon_ == sistrip::PEDESTALS_MEAN ) {
      value[0] = iter->second->pedsMean()[0];
      error[1] = iter->second->pedsSpread()[0]; 
      value[0] = iter->second->pedsMean()[1];
      error[1] = iter->second->pedsSpread()[1];
    } else if ( mon_ == sistrip::PEDESTALS_SPREAD ) { 
      value[0] = iter->second->pedsSpread()[0]; 
      value[1] = iter->second->pedsSpread()[1]; 
    } else if ( mon_ == sistrip::PEDESTALS_MAX ) { 
      value[0] = iter->second->pedsMax()[0]; 
      value[1] = iter->second->pedsMax()[1];
    } else if ( mon_ == sistrip::PEDESTALS_MIN ) { 
      value[0] = iter->second->pedsMin()[0]; 
      value[1] = iter->second->pedsMin()[1]; 
    } else if ( mon_ == sistrip::NOISE_ALL_STRIPS ) {
      uint16_t bins = noise[0].size();
      if ( noise[0].size() < noise[1].size() ) { bins = noise[1].size(); }
      for ( uint16_t inoise = 0; inoise < bins; inoise++ ) {
	value[0] = noise[0][inoise]; 
	value[1] = noise[1][inoise]; 
      }
    } else if ( mon_ == sistrip::NOISE_MEAN ) {
      value[0] = iter->second->noiseMean()[0];
      error[0] = iter->second->noiseSpread()[0]; 
      value[1] = iter->second->noiseMean()[1];
      error[1] = iter->second->noiseSpread()[1]; 
    } else if ( mon_ == sistrip::NOISE_SPREAD ) { 
      value[0] = iter->second->noiseSpread()[0]; 
      value[1] = iter->second->noiseSpread()[1]; 
    } else if ( mon_ == sistrip::NOISE_MAX ) { 
      value[0] = iter->second->noiseMax()[0]; 
      value[1] = iter->second->noiseMax()[1]; 
    } else if ( mon_ == sistrip::NOISE_MIN ) { 
      value[0] = iter->second->noiseMin()[0]; 
      value[1] = iter->second->noiseMin()[1]; 
    } else if ( mon_ == sistrip::NUM_OF_DEAD ) { 
      value[0] = 1. * iter->second->dead()[0].size(); 
      value[1] = 1. * iter->second->dead()[1].size();
    } else if ( mon_ == sistrip::NUM_OF_NOISY ) { 
      value[0] = 1. * iter->second->noisy()[0].size(); 
      value[1] = 1. * iter->second->noisy()[1].size();
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

    SummaryPlotFactoryBase::generator_->fillMap( SummaryPlotFactoryBase::level_, 
						 SummaryPlotFactoryBase::gran_, 
						 iter->first, 
						 value[1],
						 error[1] );

  }
  
  return SummaryPlotFactoryBase::generator_->nBins();

}

//------------------------------------------------------------------------------
//
void SummaryPlotFactory<PedestalsAnalysis*>::fill( TH1& summary_histo ) {
  
  // Histogram filling and formating
  SummaryPlotFactoryBase::fill( summary_histo );
  
  if ( !SummaryPlotFactoryBase::generator_ ) { return; }
  
  // Histogram formatting
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
  } else { 
    edm::LogWarning(mlSummaryPlots_)
      << "[SummaryPlotFactory::" << __func__ << "]" 
      << " Unexpected SummaryHisto value:"
      << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
  } 

}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<PedestalsAnalysis*>;

