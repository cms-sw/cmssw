#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
VpspScanTask::VpspScanTask( DQMStore* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "VpspScanTask" ),
  vpsp_()
{}

// -----------------------------------------------------------------------------
//
VpspScanTask::~VpspScanTask() {
}

// -----------------------------------------------------------------------------
//
void VpspScanTask::book() {
  
  uint16_t nbins = 60;
 
  std::string title;

  vpsp_.resize(2);
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    if ( connection().i2cAddr(iapv) ) { 

      std::stringstream extra_info; 
      extra_info << sistrip::apv_ << iapv;
      
      title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
				 sistrip::VPSP_SCAN, 
				 sistrip::FED_KEY, 
				 fedKey(),
				 sistrip::LLD_CHAN, 
				 connection().lldChannel(),
				 extra_info.str() ).title();
      
      vpsp_[iapv].histo( dqm()->bookProfile( title, title, 
					     nbins, -0.5, nbins*1.-0.5,
					     1025, 0., 1025. ) );
      
      vpsp_[iapv].vNumOfEntries_.resize(nbins,0);
      vpsp_[iapv].vSumOfContents_.resize(nbins,0);
      vpsp_[iapv].vSumOfSquares_.resize(nbins,0);
      
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void VpspScanTask::fill( const SiStripEventSummary& summary,
			 const edm::DetSet<SiStripRawDigi>& digis ) {

  // Retrieve VPSP setting and CCU channel
  uint32_t vpsp = summary.vpsp();
  uint32_t ccu_chan = summary.vpspCcuChan();

  // Check CCU channel from EventSummary is consistent with this module
  if ( SiStripFecKey( fecKey() ).ccuChan() != ccu_chan ) { return; }

  if ( digis.data.size() != 256 ) {
    edm::LogWarning(mlDqmSource_)
      << "[VpspScanTask::" << __func__ << "]"
      << " Unexpected number of digis! " 
      << digis.data.size(); 
    return;
  }

  // Fill histo with baseline(calc'ed from median value of data)
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    
    if ( vpsp >= vpsp_[iapv].vNumOfEntries_.size() ) { 
      edm::LogWarning(mlDqmSource_)
	<< "[VpspScanTask::" << __func__ << "]"
	<< " Unexpected VPSP value! " << vpsp;
      return;
    }
    
    std::vector<uint16_t> baseline;
    baseline.reserve(128); 
    for ( uint16_t idigi = 128*iapv; idigi < 128*(iapv+1); idigi++ ) {
      baseline.push_back( digis.data[idigi].adc() ); 
    }
    sort( baseline.begin(), baseline.end() ); 
    uint16_t index = baseline.size()%2 ? baseline.size()/2 : baseline.size()/2-1;
    
    if ( !baseline.empty() ) { 
      updateHistoSet( vpsp_[iapv], vpsp, baseline[index] );
    }
    
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanTask::update() {
  for ( uint32_t iapv = 0; iapv < vpsp_.size(); iapv++ ) {
    updateHistoSet( vpsp_[iapv] );
  }
}


