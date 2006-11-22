#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//@@ nBins_ should be number of scope mode samples from trigger fed data???
ApvTimingTask::ApvTimingTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "ApvTimingTask" ),
  timing_(),
  nSamples_(40),
  nFineDelays_(24), // 24 fine delays per "coarse" sample
  nBins_(nSamples_*nFineDelays_) 
{}

// -----------------------------------------------------------------------------
//
ApvTimingTask::~ApvTimingTask() {
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::book() {

  string title;
  title = SiStripHistoNamingScheme::histoTitle( HistoTitle( sistrip::APV_TIMING, 
							    sistrip::FED_KEY, 
							    fedKey(),
							    sistrip::LLD_CHAN, 
							    connection().lldChannel() ) );
  
  float min_time_ns = static_cast<float>(nBins_) - 0.5;
  float max_time_ns = (25./static_cast<float>(nFineDelays_)) * static_cast<float>(nBins_) - 0.5;
  timing_.histo_ = dqm()->bookProfile( title, title, 
				       nBins_, min_time_ns, max_time_ns,
				       sistrip::maximum_, 0., sistrip::maximum_*1. );
  
  timing_.vNumOfEntries_.resize(nBins_,0);
  timing_.vSumOfContents_.resize(nBins_,0);
  timing_.vSumOfSquares_.resize(nBins_,0);
  
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
  
  if ( digis.data.size() < nBins_ ) { // check = scope mode length? 
    edm::LogWarning(mlDqmSource_)
      << "[ApvTimingTask::" << __func__ << "]"
      << " Unexpected number of digis! " 
      << digis.data.size(); 
    return;
  }

  pair<uint32_t,uint32_t> skews = const_cast<SiStripEventSummary&>(summary).pll();
  for ( uint16_t coarse = 0; coarse < nSamples_/*digis.data.size()*/; coarse++ ) {
    uint16_t fine = (coarse+1)*nFineDelays_ - (skews.second+1);
    updateHistoSet( timing_, fine, digis.data[coarse].adc() );
  }
  
}

// -----------------------------------------------------------------------------
//
void ApvTimingTask::update() {
  updateHistoSet( timing_ );
}


