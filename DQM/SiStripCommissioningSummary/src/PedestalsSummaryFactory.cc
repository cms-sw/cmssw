#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<PedestalsAnalysis>::SummaryHistogramFactory() :
  histo_(sistrip::UNKNOWN_SUMMARY_HISTO),
  type_(sistrip::UNKNOWN_SUMMARY_TYPE),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{
} 


// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<PedestalsAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<PedestalsAnalysis>::init( const sistrip::SummaryHisto& histo, 
						       const sistrip::SummaryType& type,
						       const sistrip::View& view, 
						       const string& top_level_dir, 
						       const sistrip::Granularity& gran ) {
  histo_ = histo;
  type_ = type;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { delete generator_; generator_ = 0; }
  generator_ = SummaryGenerator::instance( view );
  
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<PedestalsAnalysis>::extract( const map<uint32_t,PedestalsAnalysis>& data ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data in monitorables map!" << endl;
    return 0; 
  }
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return 0;
  }
  
  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  map<uint32_t,PedestalsAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo_ == sistrip::PEDESTALS_ALL_STRIPS ) {
      uint16_t bins = 0;
      if ( iter->second.peds()[0].size() < 
	   iter->second.peds()[1].size() ) 
	{ bins = iter->second.peds()[0].size(); } 
      else { bins = iter->second.peds()[0].size(); }
      for ( uint16_t iped = 0; iped < bins; iped++ ) {
	generator_->fillMap( level_, gran_, iter->first, iter->second.peds()[0][iped] ); 
	generator_->fillMap( level_, gran_, iter->first, iter->second.peds()[1][iped] ); 
      }
    } else if ( histo_ == sistrip::PEDESTALS_MEAN ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.pedsMean()[0], iter->second.pedsSpread()[0] ); 
    } else if ( histo_ == sistrip::PEDESTALS_SPREAD ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.pedsSpread()[0] ); 
    } else if ( histo_ == sistrip::PEDESTALS_MAX ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.pedsMax()[0] ); 
    } else if ( histo_ == sistrip::PEDESTALS_MIN ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.pedsMin()[0] ); 
    } else if ( histo_ == sistrip::NOISE_ALL_STRIPS ) {
      uint16_t bins = 0;
      if ( iter->second.noise()[0].size() < 
	   iter->second.noise()[1].size() ) 
	{ bins = iter->second.noise()[0].size(); } 
      else { bins = iter->second.noise()[0].size(); }
      for ( uint16_t inoise = 0; inoise < bins; inoise++ ) {
	generator_->fillMap( level_, gran_, iter->first, iter->second.noise()[0][inoise] ); 
	generator_->fillMap( level_, gran_, iter->first, iter->second.noise()[1][inoise] ); 
      }
    } else if ( histo_ == sistrip::NOISE_MEAN ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.noiseMean()[0], iter->second.noiseSpread()[0] ); 
    } else if ( histo_ == sistrip::NOISE_SPREAD ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.noiseSpread()[0] ); 
    } else if ( histo_ == sistrip::NOISE_MAX ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.noiseMax()[0] ); 
    } else if ( histo_ == sistrip::NOISE_MIN ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.noiseMin()[0] ); 
    } else if ( histo_ == sistrip::NUM_OF_DEAD ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.dead().size() ); 
    } else if ( histo_ == sistrip::NUM_OF_NOISY ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.noise().size() ); 
    } else { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	   << " Unexpected SummaryHisto value:"
	   << SiStripHistoNamingScheme::summaryHisto( histo_ ) 
	   << endl;
      continue;
    }
  }

  return generator_->size();

}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<PedestalsAnalysis>::fill( TH1& summary_histo ) {

  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }

  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }

  // Check if map is filled
  if ( !generator_->size() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data in the monitorables map!" << endl;
    return; 
  } 

  // Generate appropriate summary histogram 
  if ( type_ == sistrip::SUMMARY_DISTR ) {
    generator_->summaryDistr( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_1D ) {
    generator_->summary1D( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_2D ) {
    generator_->summary2D( summary_histo );
  } else if ( type_ == sistrip::SUMMARY_PROF ) {
    generator_->summaryProf( summary_histo );
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " Unexpected SummaryType value:"
	 << SiStripHistoNamingScheme::summaryType( type_ ) 
	 << endl;
    return; 
  }
  
  // Histogram formatting
  if ( histo_ == sistrip::PEDESTALS_ALL_STRIPS ) {
    generator_->axisLabel( "Pedestal value [adc]" );
  } else if ( histo_ == sistrip::PEDESTALS_MEAN ) {
  } else if ( histo_ == sistrip::PEDESTALS_SPREAD ) { 
  } else if ( histo_ == sistrip::PEDESTALS_MAX ) { 
  } else if ( histo_ == sistrip::PEDESTALS_MIN ) { 
  } else if ( histo_ == sistrip::NOISE_ALL_STRIPS ) {
    generator_->axisLabel( "Noise [adc]" );
  } else if ( histo_ == sistrip::NOISE_MEAN ) {
  } else if ( histo_ == sistrip::NOISE_SPREAD ) { 
  } else if ( histo_ == sistrip::NOISE_MAX ) { 
  } else if ( histo_ == sistrip::NOISE_MIN ) { 
  } else if ( histo_ == sistrip::NUM_OF_DEAD ) { 
  } else if ( histo_ == sistrip::NUM_OF_NOISY ) { 
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " Unexpected SummaryHisto value:"
	 << SiStripHistoNamingScheme::summaryHisto( histo_ ) 
	 << endl;
  } 
  generator_->format( sistrip::PEDESTALS, histo_, type_, view_, level_, gran_, summary_histo );

}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<PedestalsAnalysis>;

