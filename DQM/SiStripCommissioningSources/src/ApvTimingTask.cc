#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
ApvTimingTask::ApvTimingTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "ApvTimingTask" ),
  timing_(),
  nBins_(40) //@@ this should be from number of scope mode samples (mean booking in event loop and putting scope mode length in trigger fed)
{
  edm::LogInfo("Commissioning") << "[ApvTimingTask::ApvTimingTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
ApvTimingTask::~ApvTimingTask() {
  edm::LogInfo("Commissioning") << "[ApvTimingTask::ApvTimingTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::book() {
  edm::LogInfo("Commissioning") << "[ApvTimingTask::book]";

  uint16_t nbins = 24 * nBins_; // 24 "fine" pll skews possible

  string title;
  
  title = SiStripHistoNamingScheme::histoTitle( sistrip::APV_TIMING, 
						sistrip::COMBINED, 
						sistrip::FED_KEY, 
						fedKey(),
						sistrip::LLD_CHAN, 
						connection().lldChannel() );
  
  timing_.histo_ = dqm()->bookProfile( title, title, 
				       nbins, -0.5, nBins_*25.-0.5,
				       1025, 0., 1025. );
  
  timing_.vNumOfEntries_.resize(nbins,0);
  timing_.vSumOfContents_.resize(nbins,0);
  timing_.vSumOfSquares_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
/*
  Some notes: 
  - use all samples 
  - extract number of samples from trigger fed
  - need to book histos in event loop?
  - why only use fine skew setting when filling histos? should use coarse setting as well?
  - why do different settings every 100 events - change more freq? 
*/
void ApvTimingTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[ApvTimingTask::fill]";

  //@@ if scope mode length is in trigger fed, then 
  //@@ can add check here on number of digis
  if ( digis.data.size() < nBins_ ) {
    edm::LogError("Commissioning") << "[ApvTimingTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
  } else {
    pair<uint32_t,uint32_t> skews = const_cast<SiStripEventSummary&>(summary).pll();
    for ( uint16_t coarse = 0; coarse < nBins_/*digis.data.size()*/; coarse++ ) {
      uint16_t fine = (coarse+1)*24 - (skews.second+1);
      updateHistoSet( timing_, fine, digis.data[coarse].adc() );
    }
  }

}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  LogDebug("Commissioning") << "[ApvTimingTask::update]";
  updateHistoSet( timing_ );
//   for ( uint16_t fine = 0; fine < timing_.vNumOfEntries_.size(); fine++ ) {
//     timing_.meSumOfSquares_->setBinContent( fine+1, timing_.vSumOfSquares_[fine]*1. );
//     timing_.meSumOfContents_->setBinContent( fine+1, timing_.vSumOfContents_[fine]*1. );
//     timing_.meNumOfEntries_->setBinContent( fine+1, timing_.vNumOfEntries_[fine]*1. );
//   }
}


