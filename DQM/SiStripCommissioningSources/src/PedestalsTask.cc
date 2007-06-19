#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

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
  std::string title;
  std::string extra_info;

  // Pedestals and noise histograms
  peds_.resize(2);
  nbins = 256;
  for ( uint16_t ihisto = 0; ihisto < 2; ihisto++ ) { 
    
    if      ( ihisto == 0 ) { extra_info = sistrip::pedsAndRawNoise_; } // "PedsAndRawNoise"; }
    else if ( ihisto == 1 ) { extra_info = sistrip::residualsAndNoise_; } // "ResidualsAndNoise"; }
    else { /**/ }
    
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			       sistrip::PEDESTALS, 
			       sistrip::FED_KEY, 
			       fedKey(),
			       sistrip::LLD_CHAN, 
			       connection().lldChannel(),
			       extra_info ).title();
  
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
    
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			       sistrip::PEDESTALS, 
			       sistrip::FED_KEY, 
			       fedKey(),
			       sistrip::APV, 
			       connection().i2cAddr(iapv),
			       sistrip::commonMode_ ).title();
    
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

  SiStripFecKey test( fecKey() );
  SiStripFecKey temp1(1,3,1,111,29,3);
  SiStripFecKey temp2(1,3,4,123,17,1);
  SiStripFecKey temp3(1,3,4,123,17,3);
  
  if ( digis.data.size() != peds_[0].vNumOfEntries_.size() ) {
    edm::LogWarning(mlDqmSource_)
      << "[PedestalsTask::" << __func__ << "]"
      << " Unexpected number of digis: " 
      << digis.data.size(); 
    return;
  }
  
  // Check number of digis
  uint16_t nbins = peds_[0].vNumOfEntries_.size();
  if ( digis.data.size() < nbins ) { nbins = digis.data.size(); }

  //@@ Inefficient!!!
  uint16_t napvs = nbins / 128;
  std::vector<uint16_t> cm; cm.resize(napvs,0);
  
  // Calc common mode for both APVs
  std::vector<uint16_t> adc;
  for ( uint16_t iapv = 0; iapv < napvs; iapv++ ) { 
    adc.clear(); adc.reserve(128);
    for ( uint16_t ibin = 0; ibin < 128; ibin++ ) { 
      if ( (iapv*128)+ibin < nbins ) { 
	adc.push_back( digis.data[(iapv*128)+ibin].adc() );  //@@ VIRGIN RAW DATA (MUX, APV READOUT)
      }
    }
    sort( adc.begin(), adc.end() ); 
    uint16_t index = adc.size()%2 ? adc.size()/2 : adc.size()/2-1;
    if ( !adc.empty() ) { 
      cm[iapv] = adc[index]; 
      if ( test.isEqual(temp1) || 
	   test.isEqual(temp2) || 
	   test.isEqual(temp3) ) {
	LogTrace(mlTest_)
	  << "TEST" << test << "\n"
	  << " iapv: " << iapv
	  << " adc.size(): " << adc.size()
	  << " index: " << index
	  << " adc[0]: " << adc[0]
	  << " adc[index]: " << adc[index]
	  << " adc[size()]: " << adc[adc.size()-1]
	  << " cm[iapv]: " << cm[iapv];
      }
    } else {
	LogTrace(mlTest_)
	  << "TEST" << test << "\n"
	  << " EMPTY CM VECTOR!";
    }
  }
  
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    updateHistoSet( peds_[0], ibin, digis.data[ibin].adc() ); // peds and raw noise
    updateHistoSet( peds_[1], ibin, (digis.data[ibin].adc()-cm[ibin/128]) ); // residuals and real noise
//     if ( test.isEqual(temp1) || 
// 	 test.isEqual(temp2) || 
// 	 test.isEqual(temp3) ) {
//       LogTrace(mlTest_)
// 	<< "TEST" << test << "\n"
// 	<< " ibin: " << ibin 
// 	<< " digis.data[ibin].adc(): " << digis.data[ibin].adc()
// 	<< " cm[ibin/128]: " << cm[ibin/128]
// 	<< " diff: " << digis.data[ibin].adc() - cm[ibin/128];
//     }
  }
  
  if ( cm.size() < cm_.size() ) {
    edm::LogWarning(mlDqmSource_)
      << "[PedestalsTask::" << __func__ << "]"
      << " Fewer CM values than expected: " << cm.size();
  }
  
  updateHistoSet( cm_[0], cm[0] );
  updateHistoSet( cm_[1], cm[1] );
  
}

// -----------------------------------------------------------------------------
//
void PedestalsTask::update() {
  
  // Pedestals and noise
  updateHistoSet( peds_[0] );
  updateHistoSet( peds_[1] );

  // Common mode
  updateHistoSet( cm_[0] );
  updateHistoSet( cm_[1] );

}


