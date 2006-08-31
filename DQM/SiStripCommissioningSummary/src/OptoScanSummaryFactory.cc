#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<OptoScanAnalysis::Monitorables>::generate( const sistrip::SummaryHisto& histo, 
									const sistrip::SummaryType& type,
									const sistrip::View& view, 
									const string& directory, 
									const map<uint32_t,OptoScanAnalysis::Monitorables>& data,
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
  map<uint32_t,OptoScanAnalysis::Monitorables>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    uint16_t igain = iter->second.gain_;
    if ( histo == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
      generator->fillMap( directory, iter->first, igain ); 
    } else if ( histo == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) {
      generator->fillMap( directory, iter->first, iter->second.bias_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
      generator->fillMap( directory, iter->first, iter->second.measGain_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
      generator->fillMap( directory, iter->first, iter->second.zeroLight_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_LINK_NOISE ) {
      generator->fillMap( directory, iter->first, iter->second.linkNoise_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) {
      generator->fillMap( directory, iter->first, iter->second.liftOff_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_LASER_THRESHOLD ) {
      generator->fillMap( directory, iter->first, iter->second.threshold_[igain] ); 
    } else if ( histo == sistrip::OPTO_SCAN_TICK_HEIGHT ) {
      generator->fillMap( directory, iter->first, iter->second.tickHeight_[igain] ); 
    } else { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	   << " Unexpected histogram!" << endl;
      return; 
    } 
  }
  
  // Generate appropriate summary histogram 
  if ( type == sistrip::SUMMARY_DISTR ) {
    generator->summaryDistr( summary_histo );
  } else if ( type == sistrip::SUMMARY_1D ) {
    generator->summary1D( summary_histo );
  } else { return; }



  // Histogram formatting
  generator->format( histo, type, view, directory, summary_histo );
  if ( histo == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
  } else if ( histo == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) {
  } else if ( histo == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
  } else if ( histo == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
  } else if ( histo == sistrip::OPTO_SCAN_LINK_NOISE ) {
  } else if ( histo == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) {
  } else if ( histo == sistrip::OPTO_SCAN_LASER_THRESHOLD ) {
  } else if ( histo == sistrip::OPTO_SCAN_TICK_HEIGHT ) {
  } else { return; } 
  
}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<OptoScanAnalysis::Monitorables>;

