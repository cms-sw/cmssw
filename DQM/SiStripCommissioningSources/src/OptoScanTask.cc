#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <math.h>

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <iomanip>

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

  uint16_t nbins = 50; //@@ correct?
  uint16_t gains = 4;

  std::string title;

  // Resize "histo sets"
  opto_.resize( gains );
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { opto_[igain].resize(3); }
  
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) { 
    for ( uint16_t ihisto = 0; ihisto < 3; ihisto++ ) { 
      
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
							nbins, 0.5, nbins*1.+0.5, // range is bias setting (1-50)
							1024, -0.5, 1023.5 );
      
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
	if ( opto_[gain].size() != 3 ) { opto_[gain].resize( 3 ); }
      }
      edm::LogWarning(mlDqmSource_)  
	<< "[OptoScanTask::" << __func__ << "]"
	<< " Unexpected gain value! " << gain;
    }

    if ( bias > 50 ) { return; } // only use bias settings 1-50
    
    // Find digital "0" and digital "1" levels from tick marks within scope mode data
    std::pair<float,float> digital_range;
    digital_range.first = sistrip::invalid_;
    digital_range.second = sistrip::invalid_;
    
    std::vector<float> baseline;
    float baseline_rms = 0;
    locateTicks( digis, digital_range, baseline, baseline_rms );

    uint16_t bin = bias - 1; // fill "bins" (0-49), not bias (1-50)
    
    // Digital "0"
    if ( digital_range.first < 1. * sistrip::valid_ ) {
      updateHistoSet( opto_[gain][0], bin, digital_range.first );
    }
    
    // Digital "1"
    if ( digital_range.second < 1. * sistrip::valid_ ) {
      updateHistoSet( opto_[gain][1], bin, digital_range.second );
    }
    
    // Baseline rms
    if ( baseline_rms < 1. * sistrip::valid_ ) {
      updateHistoSet( opto_[gain][2], bin, baseline_rms );
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
				std::vector<float>& baseline,
				float& baseline_rms ) {

  if (0) {

    // Trivial algo
    
    std::vector<uint16_t> adc; 
    adc.reserve( digis.data.size() ); 
    for ( uint16_t iadc = 0; iadc < digis.data.size(); iadc++ ) { adc.push_back( digis.data[iadc].adc() ); }
    sort( adc.begin(), adc.end() );

    SiStripFedKey key( digis.detId() );

    if ( digis.data.empty() ) {
      LogTrace(mlDqmSource_)
	<< "[OptoScanTask::" << __func__ << "]"
	<< " digis empty "
	<< " fed id/ch " 
	<< key.fedId() << "/" << key.feUnit() << "/" << key.feChan() <<  "/" << key.fedChannel();
      if ( adc.empty() ) {
	LogTrace(mlDqmSource_)
	  << "[OptoScanTask::" << __func__ << "]"
	  << " adc empty   "
	  << " fed id/ch " 
	  << key.fedId() << "/" << key.feUnit() << "/" << key.feChan() <<  "/" << key.fedChannel();
	if ( adc.size() != digis.data.size() ) {
	  LogTrace(mlDqmSource_)
	    << "[OptoScanTask::" << __func__ << "]"
	    << " diff size!  "
	    << " fed id/ch " 
	    << key.fedId() << "/" << key.feUnit() << "/" << key.feChan() <<  "/" << key.fedChannel();
	  return;
	}
	return;
      }
      return;
    }

    range.first  = adc.front();
    range.second = adc.back();

  } else {

    // More complicated algo

    int ttt = 1022;
    int bbb = 50;

    //   LogTrace(mlDqmSource_) << "BEGIN TEST...";
  
    // Copy ADC values and sort 
    std::vector<uint16_t> adc; 
    adc.reserve( digis.data.size() ); 
    for ( uint16_t iadc = 0; iadc < digis.data.size(); iadc++ ) { adc.push_back( digis.data[iadc].adc() ); }
    sort( adc.begin(), adc.end() );

    SiStripFedKey key( digis.detId() );
    
    //   if ( adc.front() < bbb && adc.back() < bbb ) {
    //     std::stringstream ss;
    //     ss << "TEST 00a "  
    //        << std::hex << key.key() << std::dec
    //        << " size " << adc.size()
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    

    //   if ( adc.front() > ttt && adc.back() > ttt ) {
    //     std::stringstream ss;
    //     ss << "TEST 00b "  
    //        << std::hex << key.key() << std::dec
    //        << " size " << adc.size()
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    

    //   if ( adc.front() < bbb && adc.back() > ttt ) {
    //     std::stringstream ss;
    //     ss << "TEST 00c "  
    //        << std::hex << key.key() << std::dec
    //        << " size " << adc.size()
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    

    //   if ( adc.front() < bbb || adc.back() < bbb ) {
    //     std::stringstream ss;
    //     ss << "TEST 00d "  
    //        << std::hex << key.key() << std::dec
    //        << " size " << adc.size()
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    
  
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
    uint16_t mid_range = adc.front() + ( adc.back() - adc.front() ) / 2;
  
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
  
    baseline.reserve( base.size() );
    baseline.assign( base.begin(), base.end() );
    range.first = base_mean; 
    range.second = tick_mean; 
    baseline_rms = base_rms;

    //   if ( range.first < bbb || range.second < bbb ) {
    //     std::stringstream ss;
    //     ss << "TEST 0a "  
    //        << std::hex << key.key() << std::dec
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    

    //   if ( range.first > ttt && range.second > ttt ) {
    //     std::stringstream ss;
    //     ss << "TEST 0b"  
    //        << std::hex << key.key() << std::dec
    //        << " front " << adc.front()
    //        << " back " << adc.back();
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    
  
    // Check for condition where tick mark top cannot be distinguished from baseline
    if ( !adc.empty() ) {
      //     LogTrace(mlDqmSource_)
      //       << "TEST 0";  
      if ( base.empty() || tick.empty() ) {
	//       LogTrace(mlDqmSource_)
	// 	<< "TEST 1";  
	range.first  = adc.front();
	range.second = adc.back();
	//       LogTrace(mlDqmSource_)
	// 	<< "TEST 2"  
	// 	<< " front " << adc.front()
	// 	<< " back " << adc.back();
	if ( key.key() == 0x0004c5c4 ) {
	  // 	LogTrace(mlDqmSource_)
	  // 	  << "TEST 3"  
	  // 	  << " front " << adc.front()
	  // 	  << " back " << adc.back();
	}
      }
    } else {
      edm::LogWarning(mlDqmSource_)
	<< "[OptoScanTask::" << __func__ << "]"
	<< " Found no ADC values!";
    }

    //   if ( range.first > ttt || range.second > ttt ) {
    //     std::stringstream ss;
    //     ss << "TEST 00e "  
    //        << std::hex << key.key() << std::dec
    //        << " first " << range.first
    //        << " second " << range.second;
    //     LogTrace(mlDqmSource_) << ss.str();
    //   }    

  }
 
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
