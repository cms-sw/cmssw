#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
PedestalsTask::PedestalsTask( DaqMonitorBEInterface* dqm,
			      const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "PedestalsTask" ),
  peds_(),
  cm_()
{
  LogTrace(mlDqmSource_)
    << "[PedestalsTask::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
PedestalsTask::~PedestalsTask() {
  LogTrace(mlDqmSource_)
    << "[PedestalsTask::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::book() {
  LogTrace(mlDqmSource_) << "[PedestalsTask::" << __func__ << "]";
  
  uint16_t nbins;
  string title;
  string extra_info;

  // Pedestals and noise histograms
  peds_.resize(2);
  nbins = 256;
  for ( uint16_t ihisto = 0; ihisto < 2; ihisto++ ) { 
    
    if      ( ihisto == 0 ) { extra_info = sistrip::pedsAndRawNoise_; } // "PedsAndRawNoise"; }
    else if ( ihisto == 1 ) { extra_info = sistrip::residualsAndNoise_; } // "ResidualsAndNoise"; }
    else { /**/ }
    
    title = SiStripHistoNamingScheme::histoTitle( HistoTitle( sistrip::PEDESTALS, 
							      sistrip::FED_KEY, 
							      fedKey(),
							      sistrip::LLD_CHAN, 
							      connection().lldChannel(),
							      extra_info ) );
    
    peds_[ihisto].histo_ = dqm()->bookProfile( title, title, 
					       nbins, -0.5, nbins*1.-0.5,
					       1025, 0., 1025. );
    
    peds_[ihisto].vNumOfEntries_.resize(nbins,0);
    peds_[ihisto].vSumOfContents_.resize(nbins,0);
    peds_[ihisto].vSumOfSquares_.resize(nbins,0);
    
  } 
  
  // Common mode histograms
  cm_.resize(2);
  nbins = 1024;
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) { 

    title = SiStripHistoNamingScheme::histoTitle( HistoTitle( sistrip::PEDESTALS, 
							      sistrip::FED_KEY, 
							      fedKey(),
							      sistrip::APV, 
							      connection().i2cAddr(iapv),
							      sistrip::commonMode_ ) );

    cm_[iapv].histo_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
    cm_[iapv].isProfile_ = false;
    
    cm_[iapv].vNumOfEntries_.resize(nbins,0);
    cm_[iapv].vNumOfEntries_.resize(nbins,0);
    
  }
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::fill( const SiStripEventSummary& summary,
			  const edm::DetSet<SiStripRawDigi>& digis ) {
  LogTrace(mlDqmSource_) << "[PedestalsTask::" << __func__ << "]";
  
  if ( digis.data.size() != peds_[0].vNumOfEntries_.size() ) {
    edm::LogWarning(mlDqmSource_)
      << "[PedestalsTask::" << __func__ << "]"
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
    updateHistoSet( peds_[1], ibin, (digis.data[ibin].adc()-cm[ibin/128]) ); // noise
    //     if ( digis.data[ibin].adc()-cm[ibin/128] < 0 ) {
    //       cout << "bin/apv/digi/cm/result: " 
    // 	   << ibin << " " << ibin/128 << " " 
    // 	   << digis.data[ibin].adc() << " " 
    // 	   << cm[ibin/128] << " " 
    // 	   << digis.data[ibin].adc()-cm[ibin/128]
    // 	   << endl;
    //     }
  }
  
  if ( cm.size() < cm_.size() ) {
    edm::LogWarning(mlDqmSource_)
      << "[PedestalsTask::" << __func__ << "]"
      << " Fewer CM values than expected!";
  }
  
  updateHistoSet( cm_[0], cm[0], 1 ); // (value is ignored)
  updateHistoSet( cm_[1], cm[1], 1 ); // (value is ignored)
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  LogTrace(mlDqmSource_)
    << "[PedestalsTask::" << __func__ << "]"
    << " Updating pedestal histograms for FEC key "
    << hex << setw(8) << setfill('0') << fecKey();
  
  // Pedestals and noise
  updateHistoSet( peds_[0] );
  updateHistoSet( peds_[1] );

  // Common mode
  updateHistoSet( cm_[0] );
  updateHistoSet( cm_[1] );

}


