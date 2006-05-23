#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// -----------------------------------------------------------------------------
//
PedestalsTask::PedestalsTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "PedestalsTask" ),
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
void PedestalsTask::book() {
  edm::LogInfo("Commissioning") << "[PedestalsTask::book]";
  
  uint16_t nbins = 256;
  
  string title;
  
  title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::PEDESTALS, 
						SiStripHistoNamingScheme::SUM2, 
						SiStripHistoNamingScheme::FED, 
						fedKey(),
						SiStripHistoNamingScheme::LLD_CHAN, 
						connection().lldChannel() );
  peds_.meSumOfSquares_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
  
  title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::PEDESTALS, 
						SiStripHistoNamingScheme::SUM, 
						SiStripHistoNamingScheme::FED, 
						fedKey(),
						SiStripHistoNamingScheme::LLD_CHAN, 
						connection().lldChannel() );
  peds_.meSumOfContents_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );

  title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::PEDESTALS, 
						SiStripHistoNamingScheme::NUM, 
						SiStripHistoNamingScheme::FED, 
						fedKey(),
						SiStripHistoNamingScheme::LLD_CHAN, 
						connection().lldChannel() );
  peds_.meNumOfEntries_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
  
  peds_.vSumOfSquares_.resize(nbins,0);
  peds_.vSumOfSquaresOverflow_.resize(nbins,0);
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
    updateHistoSet( peds_, ibin, digis.data[ibin].adc() );
//     peds_.vSumOfSquares_[ibin] += digis.data[ibin].adc() * digis.data[ibin].adc();
//     peds_.vSumOfContents_[ibin] += digis.data[ibin].adc();
//     peds_.vNumOfEntries_[ibin]++;
  }

}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  LogDebug("Commissioning") << "[PedestalsTask::update]";
  updateHistoSet( peds_ );
//   for ( uint16_t ibin = 0; ibin < peds_.vNumOfEntries_.size(); ibin++ ) {
//     peds_.meSumOfSquares_->setBinContent( ibin+1, peds_.vSumOfSquares_[ibin] );
//     peds_.meSumOfContents_->setBinContent( ibin+1, peds_.vSumOfContents_[ibin] );
//     peds_.meNumOfEntries_->setBinContent( ibin+1, peds_.vNumOfEntries_[ibin] );
//   }
}


