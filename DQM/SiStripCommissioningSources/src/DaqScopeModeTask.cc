#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
DaqScopeModeTask::DaqScopeModeTask( DQMStore* dqm,
				    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "DaqScopeModeTask" ),
  scope_(),
  nBins_(256) //@@ number of strips per FED channel
{}

// -----------------------------------------------------------------------------
//
DaqScopeModeTask::~DaqScopeModeTask() {
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::book() {
  LogTrace(mlDqmSource_) << "[CommissioningTask::" << __func__ << "]";
  
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 sistrip::DAQ_SCOPE_MODE, 
					 sistrip::FED_KEY, 
					 fedKey(),
					 sistrip::LLD_CHAN, 
					 connection().lldChannel() ).title();

  scope_.histo( dqm()->book1D( title, title, 
			       nBins_, -0.5, nBins_-0.5 ) );
		
  scope_.vNumOfEntries_.resize(nBins_,0);
  scope_.vSumOfContents_.resize(nBins_,0);
  scope_.vSumOfSquares_.resize(nBins_,0);
  scope_.isProfile_ = false; 
  
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::fill( const SiStripEventSummary& summary,
			     const edm::DetSet<SiStripRawDigi>& digis ) {
  
  // Only fill every 'N' events 
  if ( !updateFreq() || fillCntr()%updateFreq() ) { return; }
  
  if ( digis.data.size() != nBins_ ) { //@@ check scope mode length?  
    edm::LogWarning(mlDqmSource_)
      << "[DaqScopeModeTask::" << __func__ << "]"
      << " Unexpected number of digis (" 
      << digis.data.size()
      << ") wrt number of histogram bins ("
      << nBins_ << ")!";
  }
  
  uint16_t bins = digis.data.size() < nBins_ ? digis.data.size() : nBins_;
  for ( uint16_t ibin = 0; ibin < bins; ibin++ ) {
      updateHistoSet( scope_, ibin, digis.data[ibin].adc() ); 
  }
  
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::update() {
  updateHistoSet( scope_ );
}


