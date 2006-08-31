#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

// -----------------------------------------------------------------------------
//
VpspScanTask::VpspScanTask( DaqMonitorBEInterface* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "VpspScanTask" ),
  vpsp_()
{
  edm::LogInfo("Commissioning") << "[VpspScanTask::VpspScanTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
VpspScanTask::~VpspScanTask() {
  edm::LogInfo("Commissioning") << "[VpspScanTask::VpspScanTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
void VpspScanTask::book() {
  edm::LogInfo("Commissioning") << "[VpspScanTask::book]";
  
  uint16_t nbins = 60;
 
  string title;

  vpsp_.resize(2);
  for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
    if ( connection().i2cAddr(iapv) ) { 

      stringstream extra_info; 
      extra_info << sistrip::apv_ << iapv;
      
      title = SiStripHistoNamingScheme::histoTitle( sistrip::VPSP_SCAN, 
						    sistrip::COMBINED, 
						    sistrip::FED, 
						    fedKey(),
						    sistrip::LLD_CHAN, 
						    connection().lldChannel(),
						    extra_info.str() );

      vpsp_[iapv].histo_ = dqm()->bookProfile( title, title, 
					       nbins, -0.5, nbins*1.-0.5,
					       1024, 0., 1024. );
      
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
  LogDebug("Commissioning") << "[VpspScanTask::fill]";

  // Retrieve VPSP setting from SiStripEventSummary
  uint32_t vpsp = const_cast<SiStripEventSummary&>(summary).vpsp();
  LogDebug("Commissioning") << "[VpspScanTask::fill]" 
			    << "  VPSP: " << vpsp;
  
  if ( digis.data.size() != 256 ) {
    edm::LogError("Commissioning") << "[VpspScanTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
    return;
  } else {

    for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
      
      if ( vpsp >= vpsp_[iapv].vNumOfEntries_.size() ) { 
	edm::LogError("Commissioning") << "[VpspScanTask::fill]" 
				       << "  Unexpected VPSP value! " << vpsp;
	return;
      }

      // Determine median baseline level
      //vector<uint16_t> baseline;
      vector<uint16_t> baseline;
      baseline.reserve(128); 
      for ( uint16_t idigi = 128*iapv; idigi < 128*(iapv+1); idigi++ ) {
	baseline.push_back( digis.data[idigi].adc() ); 
      }
      sort( baseline.begin(), baseline.end() ); 
      uint16_t index = baseline.size()%2 ? baseline.size()/2 : baseline.size()/2-1;
      
      // If baseline level found, fill HistoSet vectors
      if ( !baseline.empty() ) { 
	updateHistoSet( vpsp_[iapv], vpsp, baseline[index] );
      }

    }
  }

}

// -----------------------------------------------------------------------------
//
void VpspScanTask::update() {
  LogDebug("Commissioning") << "[VpspScanTask::update]";
  for ( uint32_t iapv = 0; iapv < vpsp_.size(); iapv++ ) {
    updateHistoSet( vpsp_[iapv] );
  }
}


