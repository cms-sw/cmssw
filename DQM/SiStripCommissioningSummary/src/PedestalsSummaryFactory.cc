#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<PedestalsAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									 const sistrip::SummaryType& type,
									 const sistrip::View& view, 
									 const string& directory, 
									 const map<uint32_t,PedestalsAnalysis::Monitorables>& data,
									 TH1& summary_histo ) {

  // Check if data are present
  if ( data.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data to histogram!" << endl;
    return ; 
  } 
  
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
  if ( type == sistrip::SUMMARY_DISTR ) {
    generator->summaryDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_1D ) {
    generator->summary1D( summary_histo );
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

