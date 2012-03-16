#include "DQM/SiStripCommissioningSources/interface/NoiseHVScanTask.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <math.h>



// -----------------------------------------------------------------------------
//
NoiseHVScanTask::NoiseHVScanTask( DQMStore * dqm,
                                  const FedChannelConnection & conn,
                                  const edm::ParameterSet & pset) :
  CommissioningTask( dqm, conn, "NoiseHVScanTask" ),
  nstrips(256),
  ncmstrips(32),
  nbins(2048)
{

  LogTrace(sistrip::mlDqmSource_)
    << "[NoiseHVScanTask::" << __func__ << "]"
    << " Constructing object...";

}

// -----------------------------------------------------------------------------
//
NoiseHVScanTask::~NoiseHVScanTask() {

  LogTrace(sistrip::mlDqmSource_)
    << "[NoiseHVScanTask::" << __func__ << "]"
    << " Destructing object...";

  // process data for last HV point
  fillHVPoint(hvCurrent_);

  updateHistoSet(avgnoise_);

}

// -----------------------------------------------------------------------------
//
void NoiseHVScanTask::book() {
  LogTrace(sistrip::mlDqmSource_) << "[NoiseHVScanTask::" << __func__ << "]";

  // tracking of the HV point
  hvDone_.clear();
  hvCurrent_ = -1;

  // transient containers
  vNumOfEntries_.resize(nstrips,0);
  vSumOfContents_.resize(nstrips,0);
  vSumOfSquares_.resize(nstrips,0);

  // histogram
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
                                         sistrip::NOISE_HVSCAN, 
                                         sistrip::FED_KEY, 
                                         fedKey(),
                                         sistrip::LLD_CHAN, 
                                         connection().lldChannel(),
                                         sistrip::extrainfo::noise_ ).title();
  avgnoise_.isProfile_ = true;
  avgnoise_.histo( dqm()->bookProfile( title, title, 
                                       nbins, -0.5, nbins*1.-0.5,
                                       256, 0, 16 ) );
  avgnoise_.vNumOfEntries_.resize(nbins,0);
  avgnoise_.vSumOfContents_.resize(nbins,0);
  avgnoise_.vSumOfSquares_.resize(nbins,0);

}

// -----------------------------------------------------------------------------
//
void NoiseHVScanTask::fill( const SiStripEventSummary & summary,
                            const edm::DetSet<SiStripRawDigi> & digis ) {

  // check the HV value and take appropriate action
  uint32_t hv = summary.highVoltage();

  // check if the HV value changed
  if ((int) hv != hvCurrent_) {
    // check if we did this point already (catching events that are out of order)
    for (size_t i = 0; i != hvDone_.size(); ++i) { if (hvDone_.at(i) == hv) return; }
    // move to next HV point if this is not the first
    if (hvCurrent_ != -1) fillHVPoint(hvCurrent_);
    // set the current HV to the new one
    edm::LogInfo(sistrip::mlDqmSource_)
      << "[NoiseHVScanTask::" << __func__ << "]"
      << " Process HV point: " 
      << hv;
    hvCurrent_ = hv;
  }

  // Check number of digis
  if ( digis.data.size() != vNumOfEntries_.size() ) {
    edm::LogWarning(sistrip::mlDqmSource_)
      << "[NoiseHVScanTask::" << __func__ << "]"
      << " Unexpected number of digis: " 
      << digis.data.size(); 
    return;
  }

  // Calc common mode for both APVs
  uint16_t napvs = nstrips / 128;
  std::vector<uint32_t> cm;  cm.resize(napvs, 0);
  std::vector<uint16_t> adc;
  for ( uint16_t iapv = 0; iapv < napvs; iapv++ ) { 
    adc.clear(); adc.reserve(ncmstrips);
    for ( uint16_t ibin = 0; ibin < ncmstrips; ibin++ ) { 
      if ( (iapv*128)+ibin < nbins ) { 
        adc.push_back( digis.data[(iapv*128)+ibin].adc() );
      }
    }
    sort( adc.begin(), adc.end() ); 
    uint16_t index = adc.size()%2 ? adc.size()/2 : adc.size()/2-1;
    if ( !adc.empty() ) { cm[iapv] = static_cast<uint32_t>( adc[index] ); }
  }

  // fill the vectors with the CM-subtracted ADC counts
  for ( uint16_t istrip = 0; istrip < nstrips; istrip++ ) {
    ++ (vNumOfEntries_[istrip]);
    float value = static_cast<float>( digis.data[istrip].adc() ) - static_cast<float>( cm[istrip/128] );
    vSumOfContents_[istrip] += value;
    vSumOfSquares_[istrip] += value * value;
  }
  
}

// -----------------------------------------------------------------------------
//
void NoiseHVScanTask::update() {
}

// -----------------------------------------------------------------------------
//
void NoiseHVScanTask::fillHVPoint(uint16_t hv) {

  edm::LogInfo(sistrip::mlDqmSource_)
    << "[NoiseHVScanTask::" << __func__ << "]"
    << " Fill histogram for HV bin " << hv;

  // calculate noise per strip, average noise and noise spread
  float sumofnoise = 0, sumofsqnoise = 0;
  for ( uint16_t i = 0; i < nstrips; ++i ) {
    float mean        = (vNumOfEntries_[i] > 0 ? vSumOfContents_[i] / vNumOfEntries_[i] : 0);
    vSumOfContents_[i] = (vNumOfEntries_[i] > 0 ? sqrt( vSumOfSquares_[i] / vNumOfEntries_[i] - mean * mean ) : 0);
    sumofnoise += vSumOfContents_[i];
    sumofsqnoise += vSumOfContents_[i] * vSumOfContents_[i];
  }
  float noisemean = sumofnoise / nstrips;
  float noisespread = sqrt(sumofsqnoise / nstrips - noisemean * noisemean);

  // fill the noise histogram in case the strip is within 2 sigma from the average noise
  for ( uint16_t i = 0; i < nstrips; ++i ) {
    if (fabs(vSumOfContents_[i] - noisemean) < 2*noisespread) updateHistoSet(avgnoise_, hv, vSumOfContents_[i]);
  }

  // mark this HV point as done
  hvDone_.push_back(hv);
  vNumOfEntries_.clear();  vNumOfEntries_.resize(nstrips,0);
  vSumOfContents_.clear(); vSumOfContents_.resize(nstrips,0);
  vSumOfSquares_.clear();  vSumOfSquares_.resize(nstrips,0);
  
}
