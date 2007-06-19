#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
DaqScopeModeTask::DaqScopeModeTask( DaqMonitorBEInterface* dqm,
				    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "DaqScopeModeTask" ),
  scope_(),
  nBins_(sistrip::maximum_) // ADC range (0->1023)
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

  scope_.histo_ = dqm()->book1D( title, title, 
				 nBins_, -0.5, nBins_-0.5 );
  
  scope_.vNumOfEntries_.resize(nBins_,0);
  scope_.vSumOfContents_.resize(nBins_,0);
  scope_.vSumOfSquares_.resize(nBins_,0);
  scope_.isProfile_ = false; //@@ simple 1D histo
  
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::fill( const SiStripEventSummary& summary,
			     const edm::DetSet<SiStripRawDigi>& digis ) {

  if ( digis.data.size() == 0 ) { //@@ check scope mode length?  
    edm::LogWarning(mlDqmSource_)
      << "[DaqScopeModeTask::" << __func__ << "]"
      << " Unexpected number of digis! " 
      << digis.data.size(); 
  } else {
    for ( uint16_t ibin = 0; ibin < digis.data.size(); ibin++ ) {
      updateHistoSet( scope_, digis.data[ibin].adc() );
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::update() {
  updateHistoSet( scope_ );
}


