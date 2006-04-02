#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
PedestalsTask::PedestalsTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn ),
  peds_()
{
  edm::LogInfo("Commissioning") << "[PedestalsTask::PedestalsTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
PedestalsTask::~PedestalsTask() {
  edm::LogInfo("Commissioning") << "[PedestalsTask::PedestalsTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::book( const FedChannelConnection& conn ) {
  edm::LogInfo("Commissioning") << "[PedestalsTask::book]";
  
  uint16_t nbins = 256;

  peds_.meSumOfSquares_  = dqm_->book1D( title( "Pedestals", "sumOfSquares", conn.lldChannel() ),
					 title( "Pedestals", "sumOfSquares", conn.lldChannel() ),
					 nbins, 0., nbins*1. );
  peds_.meSumOfContents_ = dqm_->book1D( title( "Pedestals", "sumOfContents", conn.lldChannel() ),
					 title( "Pedestals", "sumOfContents", conn.lldChannel() ), 
					 nbins, 0., nbins*1. );
  peds_.meNumOfEntries_  = dqm_->book1D( title( "Pedestals", "numOfEntries", conn.lldChannel() ),
					 title( "Pedestals", "numOfEntries", conn.lldChannel() ), 
					 nbins, 0., nbins*1. );
  
  peds_.vSumOfSquares_.resize(nbins,0);
  peds_.vSumOfContents_.resize(nbins,0);
  peds_.vNumOfEntries_.resize(nbins,0);
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[PedestalsTask::fill]";

  if ( digis.data.size() != peds_.vNumOfEntries_.size() ) {
    edm::LogError("Commissioning") << "[PedestalsTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
  }

  // Check number of digis
  uint16_t nbins = peds_.vNumOfEntries_.size();
  if ( digis.data.size() < nbins ) { nbins = digis.data.size(); }

  // Fill vectors
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    peds_.vSumOfSquares_[ibin] += digis.data[ibin].adc() * digis.data[ibin].adc();
    peds_.vSumOfContents_[ibin] += digis.data[ibin].adc();
    peds_.vNumOfEntries_[ibin]++;
  }

}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  LogDebug("Commissioning") << "[PedestalsTask::update]";
  for ( uint16_t ibin = 0; ibin < peds_.vNumOfEntries_.size(); ibin++ ) {
    peds_.meSumOfSquares_->setBinContent( ibin+1, peds_.vSumOfSquares_[ibin] );
    peds_.meSumOfContents_->setBinContent( ibin+1, peds_.vSumOfContents_[ibin] );
    peds_.meNumOfEntries_->setBinContent( ibin+1, peds_.vNumOfEntries_[ibin] );
  }
}


