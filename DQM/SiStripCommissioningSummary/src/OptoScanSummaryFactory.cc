#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<OptoScanAnalysis>::SummaryHistogramFactory() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{
} 


// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<OptoScanAnalysis>::~SummaryHistogramFactory() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<OptoScanAnalysis>::init( const sistrip::Monitorable& mon, 
						      const sistrip::Presentation& pres,
						      const sistrip::View& view, 
						      const string& top_level_dir, 
						      const sistrip::Granularity& gran ) {
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { delete generator_; generator_ = 0; }
  generator_ = SummaryGenerator::instance( view );
  
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<OptoScanAnalysis>::extract( const map<uint32_t,OptoScanAnalysis>& data  ) {
  
  // Check if data are present
  if ( data.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " No data to histogram!" << endl;
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
  map<uint32_t,OptoScanAnalysis>::const_iterator iter = data.begin();
  for ( ; iter != data.end(); iter++ ) {
    uint16_t igain = iter->second.gain();
    if ( mon_ == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
      generator_->fillMap( level_, gran_, iter->first, igain ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.bias()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.measGain()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
      generator_->fillMap( level_, gran_, iter->first, iter->second.zeroLight()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_LINK_NOISE ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.linkNoise()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.liftOff()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_LASER_THRESHOLD ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.threshold()[igain] ); 
    } else if ( mon_ == sistrip::OPTO_SCAN_TICK_HEIGHT ) {
      generator_->fillMap( level_, gran_, iter->first, iter->second.tickHeight()[igain] ); 
    } else { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	   << " Unexpected SummaryHisto value:"
	   << SiStripHistoNamingScheme::monitorable( mon_ ) 
	   << endl;
      continue;
    }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<OptoScanAnalysis>::fill( TH1& summary_histo ) {

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
  if ( pres_ == sistrip::SUMMARY_HISTO ) {
    generator_->summaryHisto( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_1D ) {
    generator_->summary1D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_2D ) {
    generator_->summary2D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_PROF ) {
    generator_->summaryProf( summary_histo );
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " Unexpected SummaryType value:"
	 << SiStripHistoNamingScheme::presentation( pres_ ) 
	 << endl;
    return; 
  }
  
  // Histogram formatting
  if ( mon_ == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { 
  } else if ( mon_ == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) {
  } else if ( mon_ == sistrip::OPTO_SCAN_MEASURED_GAIN ) { 
  } else if ( mon_ == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { 
  } else if ( mon_ == sistrip::OPTO_SCAN_LINK_NOISE ) {
  } else if ( mon_ == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) {
  } else if ( mon_ == sistrip::OPTO_SCAN_LASER_THRESHOLD ) {
  } else if ( mon_ == sistrip::OPTO_SCAN_TICK_HEIGHT ) {
  } else { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " Unexpected SummaryHisto value:"
	 << SiStripHistoNamingScheme::monitorable( mon_ ) 
	 << endl;
  } 
  generator_->format( sistrip::OPTO_SCAN, mon_, pres_, view_, level_, gran_, summary_histo );

}

// -----------------------------------------------------------------------------
//
template class SummaryHistogramFactory<OptoScanAnalysis>;

