#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//@@ nBins_ should be number of scope mode samples from trigger fed data???
ApvTimingTask::ApvTimingTask( DQMStore* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "ApvTimingTask" ),
  timing_(),
  nSamples_(40),
  nFineDelays_(24),
  nBins_(40) 
{}

// -----------------------------------------------------------------------------
//
ApvTimingTask::~ApvTimingTask() {
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::book() {
  
  uint16_t nbins = 24 * nBins_;
  
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 sistrip::APV_TIMING, 
					 sistrip::FED_KEY, 
					 fedKey(),
					 sistrip::LLD_CHAN, 
					 connection().lldChannel() ).title();
  
  timing_.histo( dqm()->bookProfile( title, title, 
				     nbins, -0.5, nBins_*25.-0.5, 
				     1025, 0., 1025. ) );
		 
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
  
  if ( digis.data.size() < nBins_ ) { //@@ check scope mode length?
    edm::LogWarning(mlDqmSource_)
      << "[ApvTimingTask::" << __func__ << "]"
      << " Unexpected number of digis! " 
      << digis.data.size(); 
    return;
  }
  
  uint32_t pll_fine = summary.pllFine();
  for ( uint16_t coarse = 0; coarse < nBins_/*digis.data.size()*/; coarse++ ) {
    uint16_t fine = (coarse+1)*24 - (pll_fine+1);
    updateHistoSet( timing_, fine, digis.data[coarse].adc() );
  }
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  updateHistoSet( timing_ );
}


