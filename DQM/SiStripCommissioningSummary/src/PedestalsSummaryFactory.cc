#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
string SummaryHistogramFactory<PedestalsAnalysis::Monitorables>::name( const sistrip::SummaryHisto& histo, 
								       const sistrip::SummaryType& type,
								       const sistrip::View& view, 
								       const string& directory ) {

  stringstream ss;
  ss << sistrip::summaryHisto_ << sistrip::sep_;
  if ( histo == sistrip::PEDESTALS_MEAN ) {
    ss << sistrip::pedestalsMean_;
  } else if ( histo == sistrip::PEDESTALS_SPREAD ) { 
    ss << sistrip::pedestalsSpread_;
  } else if ( histo == sistrip::PEDESTALS_MAX ) { 
    ss << sistrip::pedestalsMax_;
  } else if ( histo == sistrip::PEDESTALS_MIN ) { 
    ss << sistrip::pedestalsMin_;
  } else if ( histo == sistrip::NOISE_MEAN ) {
    ss << sistrip::noiseMean_;
  } else if ( histo == sistrip::NOISE_SPREAD ) { 
    ss << sistrip::noiseSpread_;
  } else if ( histo == sistrip::NOISE_MAX ) { 
    ss << sistrip::noiseMax_;
  } else if ( histo == sistrip::NOISE_MIN ) { 
    ss << sistrip::noiseMin_;
  } else if ( histo == sistrip::NUM_OF_DEAD ) { 
    ss << sistrip::numOfDead_;
  } else if ( histo == sistrip::NUM_OF_NOISY ) { 
    ss << sistrip::numOfNoisy_;
  } else { 
    ss << sistrip::unknownSummaryHisto_;
  } 
  ss << sistrip::sep_ << SiStripHistoNamingScheme::view( view );
  return ss.str(); 
  
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<PedestalsAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									 const sistrip::SummaryType& type,
									 const sistrip::View& view, 
									 const string& directory, 
									 const map<uint32_t,PedestalsAnalysis::Monitorables>& data,
									 TH1& summary_histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check if data are present
  if ( data.empty() ) { return ; } 
  
  // Retrieve utility class used to generate summary histograms
  auto_ptr<SummaryGenerator> generator = SummaryGenerator::instance( view );
  if ( !generator.get() ) { return; }

  // Transfer appropriate info from monitorables map to generator object
  map<uint32_t,PedestalsAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    if ( histo == sistrip::PEDESTALS_MEAN ) {
      generator->fillMap( directory, iter->first, iter->second.pedsMean_[0], iter->second.pedsSpread_[0] ); 
    } else if ( histo == sistrip::PEDESTALS_SPREAD ) { 
      generator->fillMap( directory, iter->first, iter->second.pedsSpread_[0] ); 
    } else if ( histo == sistrip::PEDESTALS_MAX ) { 
      generator->fillMap( directory, iter->first, iter->second.pedsMax_[0] ); 
    } else if ( histo == sistrip::PEDESTALS_MIN ) { 
      generator->fillMap( directory, iter->first, iter->second.pedsMin_[0] ); 
    } else if ( histo == sistrip::NOISE_MEAN ) {
      generator->fillMap( directory, iter->first, iter->second.noiseMean_[0], iter->second.noiseSpread_[0] ); 
    } else if ( histo == sistrip::NOISE_SPREAD ) { 
      generator->fillMap( directory, iter->first, iter->second.noiseSpread_[0] ); 
    } else if ( histo == sistrip::NOISE_MAX ) { 
      generator->fillMap( directory, iter->first, iter->second.noiseMax_[0] ); 
    } else if ( histo == sistrip::NOISE_MIN ) { 
      generator->fillMap( directory, iter->first, iter->second.noiseMin_[0] ); 
    } else if ( histo == sistrip::NUM_OF_DEAD ) { 
      generator->fillMap( directory, iter->first, iter->second.dead_.size() ); 
    } else if ( histo == sistrip::NUM_OF_NOISY ) { 
      generator->fillMap( directory, iter->first, iter->second.noise_.size() ); 
    } else { return; } 
  }

  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_SIMPLE_DISTR ) {
    generator->simpleDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_LOGICAL_VIEW ) {
    generator->logicalView( summary_histo );
  } else { return; }

  // Histogram formatting
  generator->format( histo, type, view, directory, summary_histo );
//   summary_histo.SetName( name( histo, type, view, directory ).c_str() );
//   summary_histo.SetTitle( name( histo, type, view, directory ).c_str() );
//   generator->format( summary_histo );
  if ( histo == sistrip::PEDESTALS_MEAN ) {
  } else if ( histo == sistrip::PEDESTALS_SPREAD ) { 
  } else if ( histo == sistrip::PEDESTALS_MAX ) { 
  } else if ( histo == sistrip::PEDESTALS_MIN ) { 
  } else if ( histo == sistrip::NOISE_MEAN ) {
  } else if ( histo == sistrip::NOISE_SPREAD ) { 
  } else if ( histo == sistrip::NOISE_MAX ) { 
  } else if ( histo == sistrip::NOISE_MIN ) { 
  } else if ( histo == sistrip::NUM_OF_DEAD ) { 
  } else if ( histo == sistrip::NUM_OF_NOISY ) { 
  } else { return; } 

}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<PedestalsAnalysis::Monitorables>;

