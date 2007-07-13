#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <math.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
OptoScanTask::OptoScanTask( DaqMonitorBEInterface* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "OptoScanTask" ),
  opto_()
{}

// -----------------------------------------------------------------------------
//
OptoScanTask::~OptoScanTask() {
}

// -----------------------------------------------------------------------------
//
void OptoScanTask::book() {

  uint16_t nbins = 51; //@@ correct?
  uint16_t gains = 4;

  std::string title;

  // Resize "histo sets"
  opto_.resize( gains );
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { opto_[igain].resize(3); }
  
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { 
    for ( uint16_t ihisto = 0; ihisto < 2; ihisto++ ) { 
      
      // Extra info
      std::stringstream extra_info; 
      extra_info << sistrip::gain_ << igain;
      if ( ihisto == 0 || ihisto == 1 ) {
	extra_info << sistrip::digital_ << ihisto;
      } else {
	extra_info << sistrip::baselineRms_;
      }

      // Title
      title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
				 sistrip::OPTO_SCAN, 
				 sistrip::FED_KEY, 
				 fedKey(),
				 sistrip::LLD_CHAN, 
				 connection().lldChannel(),
				 extra_info.str() ).title();

      // Book histo
      opto_[igain][ihisto].histo_ = dqm()->bookProfile( title, title, 
							nbins, -0.5, nbins*1.-0.5,
							1025, 0., 1025. );
      
      opto_[igain][ihisto].vNumOfEntries_.resize(nbins,0);
      opto_[igain][ihisto].vSumOfContents_.resize(nbins,0);
      opto_[igain][ihisto].vSumOfSquares_.resize(nbins,0);
      
    }
  }
  
}

// -----------------------------------------------------------------------------
//
/*
  improve finding tick mark and baseline
  keep baseline "noise" along with tick hieght and base
  new histo plotting noise +/- spread in noise vs bias
*/
void OptoScanTask::fill( const SiStripEventSummary& summary,
			 const edm::DetSet<SiStripRawDigi>& digis ) {

  //@@ if scope mode length is in trigger fed, then 
  //@@ can add check here on number of digis
  if ( digis.data.empty() ) {
    edm::LogWarning(mlDqmSource_)
      << "[OptoScanTask::" << __func__ << "]"
      << " Unexpected number of digis! " 
      << digis.data.size(); 
  } else {
    
    // Retrieve opt bias and gain setting from SiStripEventSummary
    uint16_t gain = summary.lldGain();
    uint16_t bias = summary.lldBias();
    if ( gain >= opto_.size() ) { 
      opto_.resize( gain );
      for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { 
	if ( opto_[gain].size() != 2 ) { opto_[gain].resize( 2 ); }
      }
      edm::LogWarning(mlDqmSource_)  
	<< "[OptoScanTask::" << __func__ << "]"
	<< " Unexpected gain value! " << gain;
    }
    
    // Find digital "0" and digital "1" levels from tick marks within scope mode data
    std::pair<float,float> digital_range;
    std::vector<float> baseline;
    locateTicks( digis, digital_range, baseline );
    
    // Digital "0"
    if ( digital_range.first < 1. * sistrip::valid_ ) {
      updateHistoSet( opto_[gain][0], bias, digital_range.first );
    }
    
    // Digital "1"
    if ( digital_range.second < 1. * sistrip::valid_ ) {
      updateHistoSet( opto_[gain][1], bias, digital_range.second );
    }
    
    // Baseline rms
    if ( !baseline.empty() ) {
      std::vector<float>::const_iterator iter = baseline.begin();
      for ( ; iter != baseline.end(); iter++ ) {
	updateHistoSet( opto_[gain][2], bias, *iter );
      }
    }
    
  }

}


// -----------------------------------------------------------------------------
//
void OptoScanTask::update() {
  
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { 
    for ( uint16_t ihisto = 0; ihisto < opto_[igain].size(); ihisto++ ) { 
      updateHistoSet( opto_[igain][ihisto] );
    }
  }

}

// -----------------------------------------------------------------------------
//
void OptoScanTask::locateTicks( const edm::DetSet<SiStripRawDigi>& digis, 
				std::pair<float,float>& range, 
				std::vector<float>& baseline ) {
  
  // Copy ADC values and sort 
  std::vector<uint16_t> adc; 
  adc.reserve( digis.data.size() ); 
  for ( uint16_t iadc = 0; iadc < digis.data.size(); iadc++ ) { adc.push_back( digis.data[iadc].adc() ); }
  sort( adc.begin(), adc.end() );
    
  // Initialization for "baseline" 
  std::vector<float> base;
  base.reserve( adc.size() );
  float base_mean = 0.;
  float base_rms = 0.;
  float base_median = 0.;
  
  // Initialization for "tick marks" 
  std::vector<float> tick;
  tick.reserve( adc.size() );
  float tick_mean = 0.;
  float tick_rms = 0.;
  float tick_median = 0.;
  
  // Calculate mid-range of data 
  uint16_t mid_range = adc.front() + ( adc.back() + adc.front() ) / 2;
  
  // Associate ADC values with either "ticks" or "baseline"
  std::vector<uint16_t>::const_iterator iter = adc.begin();
  std::vector<uint16_t>::const_iterator jter = adc.end();
  for ( ; iter != jter; iter++ ) { 
    if ( *iter < mid_range ) {
      base.push_back( *iter ); 
      base_mean += *iter;
      base_rms += (*iter) * (*iter);
    } else {
      tick.push_back( *iter ); 
      tick_mean += *iter;
      tick_rms += (*iter) * (*iter);
    }
  }

  // Calc mean and rms of baseline
  if ( !base.empty() ) { 
    base_mean = base_mean / base.size();
    base_rms = base_rms / base.size();
    base_rms = sqrt( fabs( base_rms - base_mean*base_mean ) ); 
  } else { 
    base_mean = 1. * sistrip::invalid_; 
    base_rms = 1. * sistrip::invalid_; 
    base_median = 1. * sistrip::invalid_; 
  }

  // Calc mean and rms of tick marks
  if ( !tick.empty() ) { 
    tick_mean = tick_mean / tick.size();
    tick_rms = tick_rms / tick.size();
    tick_rms = sqrt( fabs( tick_rms - tick_mean*tick_mean ) ); 
  } else { 
    tick_mean = 1. * sistrip::invalid_; 
    tick_rms = 1. * sistrip::invalid_; 
    tick_median = 1. * sistrip::invalid_; 
  }
  
  range.first = base_mean;
  range.second = tick_mean;
  copy( base.begin(), base.end(), baseline.begin() );

}

// -----------------------------------------------------------------------------
//
void OptoScanTask::deprecated( const edm::DetSet<SiStripRawDigi>& digis, 
			       std::pair< uint16_t, uint16_t >& digital_range, 
			       bool first_tick_only ) {
  
  //@@ RUBBISH!!!! simplify!!! find min/max, mid-range and push back into base and tick vectors (based on mid-range)
  
  // Copy ADC values and sort
  std::vector<uint16_t> adc; adc.reserve( digis.data.size() ); 
  for ( uint16_t iadc = 0; iadc < digis.data.size(); iadc++ ) { adc.push_back( digis.data[iadc].adc() ); }
  sort( adc.begin(), adc.end() );
  uint16_t size = adc.size();

  // Find mean and error
  float sum = 0, sum2 = 0, num = 0;
  for ( uint16_t iadc = static_cast<uint16_t>( 0.1*size ); 
	iadc < static_cast<uint16_t>( 0.9*adc.size() ); iadc++ ) {
    sum  += adc[iadc];
    sum2 += adc[iadc] * adc[iadc];
    num++;
  }
  float mean = 0, mean2 = 0, sigma = 0;
  if ( num ) { 
    mean = sum / num; 
    mean2 = sum2 / num;
    if ( mean2 > mean*mean ) { sigma = sqrt( mean2 - mean*mean ); }
  }

  // Identify samples belonging to "baseline" and "tick marks"
  float threshold = mean + sigma * 5.;
  //bool found_first_tick = false;
  std::vector<uint32_t> baseline; 
  std::vector<uint32_t> tickmark; 
  for ( uint16_t iadc = 0; iadc < adc.size(); iadc++ ) { 
    if ( adc[iadc] > threshold ) { tickmark.push_back( adc[iadc] ); }
    else { baseline.push_back( adc[iadc] ); }
  }

  // Define digital "0" and "1" levels (median values)
  if ( !baseline.empty() ) { 
    uint16_t sample = baseline.size()%2 ? baseline.size()/2 : baseline.size()/2-1;
    digital_range.first = baseline[ sample ]; // median
  } else { digital_range.first = 1025; }

  if ( !tickmark.empty() ) { 
    uint16_t sample = tickmark.size()%2 ? tickmark.size()/2 : tickmark.size()/2-1;
    digital_range.second = tickmark[ sample ]; // median
  } else { digital_range.second = 1025; }
  
}
