#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
ApvTimingTask::ApvTimingTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn ),
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
void ApvTimingTask::book( const FedChannelConnection& conn ) {
  edm::LogInfo("Commissioning") << "[ApvTimingTask::book]";

  uint16_t nbins = 24 * nBins_; // 24 "fine" pll skews possible

  timing_.meSumOfSquares_  = dqm_->book1D( title( "ApvTiming", "sumOfSquares", conn.lldChannel() ),
					   title( "ApvTiming", "sumOfSquares", conn.lldChannel() ),
					   nbins, 0., nbins*1. );
  timing_.meSumOfContents_ = dqm_->book1D( title( "ApvTiming", "sumOfContents", conn.lldChannel() ),
					   title( "ApvTiming", "sumOfContents", conn.lldChannel() ), 
					   nbins, 0., nbins*1. );
  timing_.meNumOfEntries_  = dqm_->book1D( title( "ApvTiming", "numOfEntries", conn.lldChannel() ),
					   title( "ApvTiming", "numOfEntries", conn.lldChannel() ), 
					   nbins, 0., nbins*1. );
  
  timing_.vSumOfSquares_.resize(nbins,0);
  timing_.vSumOfContents_.resize(nbins,0);
  timing_.vNumOfEntries_.resize(nbins,0);
  
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
  if ( digis.data.size() != 280 ) {
    edm::LogError("Commissioning") << "[ApvTimingTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
  }
  
  pair<uint32_t,uint32_t> skews = const_cast<SiStripEventSummary&>(summary).pll();
  //cout << "skews " << summary.event() << " " << summary.bx() << " " << skews.first << " " << skews.second << endl;
  
  // Fill vectors
  for ( uint16_t coarse = 0; coarse < nBins_/*digis.data.size()*/; coarse++ ) {
    //uint16_t fine = ( coarse + 1 ) * 25 - static_cast<uint16_t>( skews.second * 25./24. ); //@@ check formula!
    uint16_t fine = (coarse+1)*24 - skews.second; //@@ check formula!
    //cout << "fine " << coarse << " " << fine << endl;
    timing_.vSumOfSquares_[fine] += digis.data[coarse].adc() * digis.data[coarse].adc();
    timing_.vSumOfContents_[fine] += digis.data[coarse].adc();
    timing_.vNumOfEntries_[fine]++;
  }      
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  LogDebug("Commissioning") << "[ApvTimingTask::update]";
  for ( uint16_t fine = 0; fine < timing_.vNumOfEntries_.size(); fine++ ) {
    timing_.meSumOfSquares_->setBinContent( fine+1, timing_.vSumOfSquares_[fine]*1. );
    timing_.meSumOfContents_->setBinContent( fine+1, timing_.vSumOfContents_[fine]*1. );
    timing_.meNumOfEntries_->setBinContent( fine+1, timing_.vNumOfEntries_[fine]*1. );
  }
}


