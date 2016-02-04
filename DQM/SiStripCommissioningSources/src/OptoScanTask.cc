#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <math.h>

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <iomanip>


using namespace sistrip;


// -----------------------------------------------------------------------------
//
OptoScanTask::OptoScanTask( DQMStore * dqm,
			    const FedChannelConnection & conn ) :
  CommissioningTask( dqm, conn, "OptoScanTask" ),
  opto_() {
}


// -----------------------------------------------------------------------------
//
OptoScanTask::~OptoScanTask() {
}


// -----------------------------------------------------------------------------
//
void OptoScanTask::book() {

  uint16_t nbins = 50;
  uint16_t gains = 4;

  // Resize "histo sets"
  opto_.resize( gains );
  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) {
    opto_[igain].resize(3);
  }

  for ( uint16_t igain = 0; igain < opto_.size(); igain++ ) {
    for ( uint16_t ihisto = 0; ihisto < 3; ihisto++ ) {

      // Extra info
      std::stringstream extra_info;
      extra_info << sistrip::extrainfo::gain_ << igain;
      if ( ihisto == 0 || ihisto == 1 ) {
        extra_info << sistrip::extrainfo::digital_ << ihisto;
      } else {
        extra_info << sistrip::extrainfo::baselineRms_;
      }

      // Title
      std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                             sistrip::OPTO_SCAN, 
                                             sistrip::FED_KEY, 
                                             fedKey(),
                                             sistrip::LLD_CHAN, 
                                             connection().lldChannel(),
                                             extra_info.str() ).title();

      // Book histo
      opto_[igain][ihisto].histo( dqm()->bookProfile( title, title, 
                                                      nbins, 0.5, nbins*1.+0.5, // range is bias setting (1-50)
                                                      1024, -0.5, 1023.5 ) );

      opto_[igain][ihisto].vNumOfEntries_.resize(nbins,0);
      opto_[igain][ihisto].vSumOfContents_.resize(nbins,0);
      opto_[igain][ihisto].vSumOfSquares_.resize(nbins,0);

    } // end loop on histos
  } // end loop on gains

}


// -----------------------------------------------------------------------------
//
void OptoScanTask::fill( const SiStripEventSummary & summary,
			 const edm::DetSet<SiStripRawDigi> & digis ) {

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
    std::vector<float> baseline;
    std::pair<float,float> digital_range;
    digital_range.first  = sistrip::invalid_;
    digital_range.second = sistrip::invalid_;
    float baseline_rms   = sistrip::invalid_;

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
void OptoScanTask::locateTicks( const edm::DetSet<SiStripRawDigi> & digis, 
				std::pair<float,float> & range, 
				std::vector<float> & baseline,
				float & baseline_rms ) {

  // Copy ADC values and sort 
  std::vector<uint16_t> adc; 
  adc.reserve( digis.data.size() ); 
  for ( uint16_t iadc = 0; iadc < digis.data.size(); iadc++ ) {
    // only take the adc from the first APV (multiplexing alternates the digis)
    // this was asked by Karl et al
    if (iadc % 2 == 0) {
      adc.push_back( digis.data[iadc].adc() );
     }
  }
  sort( adc.begin(), adc.end() );

  // To make sure we have a tickmark, which comes every 70 bxs,
  // fully contained in the scope 'frame' we are analyzing
  // standard length is 280, sufficient to contain a full apv frame
  // in this run there is no frame though, just baseline and tickmarks
  if ( adc.size() > 70 ) {

    // Define tick mark top" level as "max" ADC values
    range.second = adc.back();

    // Construct vector to hold "baseline samples" (exclude tick mark samples)
    std::vector<uint16_t> truncated; 
    std::vector<uint16_t>::const_iterator ii = adc.begin();
    // remove twice the expected number of tick samples, otherwise you bias the baseline mean and rms
    std::vector<uint16_t>::const_iterator jj = adc.end() - 4 * ( ( adc.size() / 70 ) + 1 );
    truncated.resize( jj - ii );
    std::copy( ii, jj, truncated.begin() );
    if ( truncated.empty() ) { return; }

    // Calc mean baseline level
    float b_mean = 0.;
    std::vector<uint16_t>::const_iterator iii = truncated.begin();
    std::vector<uint16_t>::const_iterator jjj = truncated.end();
    for ( ; iii != jjj; ++iii ) { b_mean += *iii; }
    b_mean /= ( 1. * truncated.size() );
    range.first = b_mean;

    // Calc baseline noise
    float b_rms = 0.;
    std::vector<uint16_t>::const_iterator iiii = truncated.begin();
    std::vector<uint16_t>::const_iterator jjjj = truncated.end();
    for ( ; iiii != jjjj; ++iiii ) { b_rms += fabs( *iiii - b_mean ); }
    // Set baseline "noise" (requires any possible APV frames are filtered from the data!)
    baseline_rms = sqrt ( b_rms / ( 1. * truncated.size() ) );

  } else {
    edm::LogWarning(mlDqmSource_)
      << "[OptoScanTask::" << __func__ << "]"
      << " Insufficient ADC values: " << adc.size();
  }

}
