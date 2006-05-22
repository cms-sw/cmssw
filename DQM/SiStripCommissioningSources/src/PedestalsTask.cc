#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

// -----------------------------------------------------------------------------
//
PedestalsTask::PedestalsTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "PedestalsTask" ),
  peds_(),
  vCommonMode0_(),
  vCommonMode1_(),
  meCommonMode0_(0),
  meCommonMode1_(0)
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
  string extra;

  // Pedestals and noise histograms
  peds_.resize(2);
  for ( uint16_t ihisto = 0; ihisto < 2; ihisto++ ) { 
    
    if      ( ihisto == 0 ) { extra = sistrip::pedsAndRawNoise_; } // "PedsAndRawNoise"; }
    else if ( ihisto == 1 ) { extra = sistrip::residualsAndNoise_; } // "ResidualsAndNoise"; }
    else { /**/ }
    
    title = SiStripHistoNamingScheme::histoTitle( sistrip::PEDESTALS, 
						  sistrip::SUM2, 
						  sistrip::FED, 
						  fedKey(),
						  sistrip::LLD_CHAN, 
						  connection().lldChannel(),
						  extra );
    peds_[ihisto].meSumOfSquares_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
  
    title = SiStripHistoNamingScheme::histoTitle( sistrip::PEDESTALS, 
						  sistrip::SUM, 
						  sistrip::FED, 
						  fedKey(),
						  sistrip::LLD_CHAN, 
						  connection().lldChannel(),
						  extra );
    peds_[ihisto].meSumOfContents_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
  
    title = SiStripHistoNamingScheme::histoTitle( sistrip::PEDESTALS, 
						  sistrip::NUM, 
						  sistrip::FED, 
						  fedKey(),
						  sistrip::LLD_CHAN, 
						  connection().lldChannel(),
						  extra );
    peds_[ihisto].meNumOfEntries_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );

    peds_[ihisto].vSumOfSquares_.resize(nbins,0);
    peds_[ihisto].vSumOfSquaresOverflow_.resize(nbins,0);
    peds_[ihisto].vSumOfContents_.resize(nbins,0);
    peds_[ihisto].vNumOfEntries_.resize(nbins,0);
    
  } 
 
  // Common mode histograms
  nbins = 1024;
  title = SiStripHistoNamingScheme::histoTitle( sistrip::PEDESTALS, 
						sistrip::COMBINED, 
						sistrip::FED, 
						fedKey(),
						sistrip::APV, 
						connection().i2cAddr(0),
						sistrip::commonMode_ );
  meCommonMode0_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );

  title = SiStripHistoNamingScheme::histoTitle( sistrip::PEDESTALS, 
						sistrip::COMBINED, 
						sistrip::FED, 
						fedKey(),
						sistrip::APV, 
						connection().i2cAddr(1),
						sistrip::commonMode_ );
  meCommonMode1_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
  
  vCommonMode0_.resize(nbins,0);
  vCommonMode1_.resize(nbins,0);

}

// -----------------------------------------------------------------------------
//
void PedestalsTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[PedestalsTask::fill]";

  if ( digis.data.size() != peds_[0].vNumOfEntries_.size() ) {
    edm::LogError("Commissioning") << "[PedestalsTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
  }

  // Check number of digis
  uint16_t nbins = peds_[0].vNumOfEntries_.size();
  if ( digis.data.size() < nbins ) { nbins = digis.data.size(); }

  //@@ Inefficient!!!
  uint16_t napvs = nbins / 128;
  vector<uint16_t> cm; cm.resize(napvs,0);
  
  // Calc common mode for both APVs
  vector<uint16_t> adc;
  for ( uint16_t iapv = 0; iapv < napvs; iapv++ ) { 
    adc.clear(); adc.reserve(128);
    for ( uint16_t ibin = 0; ibin < 128; ibin++ ) { 
      if ( (iapv*128)+ibin < nbins ) { 
	adc.push_back( digis.data[(iapv*128)+ibin].adc() ); 
// 	cout << "ibin: " << ibin 
// 	     << " str: " << (iapv*128)+ibin 
// 	     << " size: " << adc.size()
// 	     << " adc: " << digis.data[(iapv*128)+ibin].adc() 
// 	     << " back: " << adc.back() << endl;
      }
    }
    sort( adc.begin(), adc.end() ); 
    uint16_t index = adc.size()%2 ? adc.size()/2 : adc.size()/2-1;
    if ( !adc.empty() ) { cm[iapv] = adc[index]; }
//     cout << adc.empty() << " " 
// 	 << adc.size() << " " << index << " " 
// 	 << adc[index] << " " 
// 	 << iapv << " " << cm[iapv] << endl;
  }
  
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    updateHistoSet( peds_[0], ibin, digis.data[ibin].adc() ); // peds
//     if ( digis.data[ibin].adc()-cm[ibin/128] < 0 ) {
//       cout << "bin/apv/digi/cm/result: " 
// 	   << ibin << " " << ibin/128 << " " 
// 	   << digis.data[ibin].adc() << " " 
// 	   << cm[ibin/128] << " " 
// 	   << digis.data[ibin].adc()-cm[ibin/128]
// 	   << endl;
//     }
    updateHistoSet( peds_[1], ibin, (digis.data[ibin].adc()-cm[ibin/128]) ); // noise
  }
  
  if ( vCommonMode0_.size() > cm[0] ) { vCommonMode0_[cm[0]]++; }
  if ( vCommonMode1_.size() > cm[1] ) { vCommonMode1_[cm[1]]++; }
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  LogDebug("Commissioning") << "[PedestalsTask::update]"
			    << " Updating pedestal histograms for FEC key "
			    << hex << setw(8) << setfill('0') << fecKey();
  
  // Pedestals and noise
  updateHistoSet( peds_[0] );
  updateHistoSet( peds_[1] );
  
  // Common mode
  if ( !meCommonMode0_ || 
       !meCommonMode1_ ) {
    edm::LogError("Commissioning") << "[PedestalsTask::update]" 
				   << " NULL pointer to ME!";
    return;
  }
  for ( uint32_t ibin = 0; ibin < vCommonMode0_.size(); ibin++ ) {
    meCommonMode0_->setBinContent( ibin+1, vCommonMode0_[ibin]*1. );
    meCommonMode1_->setBinContent( ibin+1, vCommonMode1_[ibin]*1. );
  }

}


