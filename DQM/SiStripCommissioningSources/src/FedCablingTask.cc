#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

// -----------------------------------------------------------------------------
//
FedCablingTask::FedCablingTask( DaqMonitorBEInterface* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "FedCablingTask" ),
  cabling_()
{
  edm::LogInfo("Commissioning") << "[FedCablingTask::FedCablingTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
FedCablingTask::~FedCablingTask() {
  edm::LogInfo("Commissioning") << "[FedCablingTask::FedCablingTask] Constructing object...";
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::book() {
  edm::LogInfo("Commissioning") << "[FedCablingTask::book]";
  
  cabling_.resize(2);
  
  string title;
  uint16_t nbins = 0;
  string extra_info = "";
  for ( uint16_t iter = 0; iter < 2; iter++ ) {
    
    // Define number of histo bins and title
    if ( iter == 0 )      { nbins = 1024; extra_info = sistrip::fedId_; }
    else if ( iter == 1 ) { nbins = 96;   extra_info = sistrip::fedChannel_; }
    else {
      edm::LogError("Commissioning") << "[FedCablingTask::book]"
				     << " Unexpected number of HistoSets" << iter;
    }
    
    title = SiStripHistoNamingScheme::histoTitle( sistrip::FED_CABLING,
						  sistrip::COMBINED, 
						  sistrip::FED_KEY, 
						  fedKey(),
						  sistrip::LLD_CHAN, 
						  connection().lldChannel(),
						  extra_info );

    cabling_[iter].histo_ = dqm()->bookProfile( title, title, 
						nbins, -0.5, nbins*1.-0.5,
						1025, 0., 1025. );
    
    cabling_[iter].vNumOfEntries_.resize(nbins,0);
    cabling_[iter].vSumOfContents_.resize(nbins,0);
    cabling_[iter].vSumOfSquares_.resize(nbins,0);
    //cabling_[iter].isProfile_ = false; //@@ using simple 1D histos
    
  }
  
}

// // -----------------------------------------------------------------------------
// //
// void FedCablingTask::fill( const SiStripEventSummary& summary,
// 			   const edm::DetSet<SiStripRawDigi>& digis ) {
//   LogDebug("Commissioning") << "[FedCablingTask::fill]";
  
//   stringstream ss; 
//   ss << "[FedCablingTask::fill] DeviceId: " 
//      << setfill('0') << setw(8) << hex << summary.deviceId() << dec;
//   LogDebug("Commissioning") << ss.str();
  
//   //@@ if scope mode length is in trigger fed, then 
//   //@@ can add check here on number of digis
//   if ( digis.data.empty() ) {
//     edm::LogError("Commissioning") << "[FedCablingTask::fill]" 
// 				   << " Unexpected number of digis! " 
// 				   << digis.data.size(); 
//   } else {
    
//     // Determine ADC median level
//     vector<uint16_t> level;
//     level.reserve(128); 
//     for ( uint16_t idigi = 0; idigi < digis.data.size(); idigi++ ) { level.push_back( digis.data[idigi].adc() ); }
//     sort( level.begin(), level.end() ); 
//     uint16_t index = level.size()%2 ? level.size()/2 : level.size()/2-1;
    
// #ifdef TEST
//     if ( connection().fedId() == fedId() &&
// 	 connection().fedCh() == fedCh() ) { 
//       level.resize(1,1000); 
//       index = 0; 
//     } else { 
//       level.resize(1,100); 
//       index = 0; 
//     }
// #endif
    
//     // Fill FED id and channel histograms
//     if ( !level.empty() ) {
//       if ( fedId() < cabling_[0].vNumOfEntries_.size() && 
// 	   fedCh() < cabling_[1].vNumOfEntries_.size() ) { 
// 	updateHistoSet( cabling_[0], fedId(), level[index] );
// 	updateHistoSet( cabling_[1], fedCh(), level[index] );
//       } else {
// 	edm::LogError("Commissioning") << "[FedCablingTask::fill]" 
// 				       << " Unexpected FED id and/or channel " << fedId() << "/" << fedCh();
// 	return;
//       }
//     }

//   }
  
// }

// -----------------------------------------------------------------------------
//
void FedCablingTask::fill( const SiStripEventSummary& summary,
			   const uint16_t& fed_id,
			   const map<uint16_t,float>& fed_ch ) {
  LogDebug("Commissioning") << "[FedCablingTask::fill]";
  
  stringstream ss; 
  ss << "[FedCablingTask::fill] DeviceId: " 
     << setfill('0') << setw(8) << hex << summary.deviceId() << dec;
  LogDebug("Commissioning") << ss.str();
  
  if ( fed_ch.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " No FED channels with high signal!" << endl;
    return; 
  }
  
  // Fill FED id and channel histogram
  //cout << "Number of FED channels found: " << fed_ch.size() << endl;
  map<uint16_t,float>::const_iterator ichan = fed_ch.begin();
  for ( ; ichan != fed_ch.end(); ichan++ ) {
//     cout << " FED id: " << fed_id
// 	 << " FED ch: " << ichan->first
// 	 << " median: " << ichan->second
// 	 << endl;
    updateHistoSet( cabling_[0], fed_id, ichan->second );
    updateHistoSet( cabling_[1], ichan->first, ichan->second );
  } 

}

// -----------------------------------------------------------------------------
//
void FedCablingTask::update() {
  LogDebug("Commissioning") << "[FedCablingTask::update]";
  for ( uint32_t iter = 0; iter < cabling_.size(); iter++ ) {
    updateHistoSet( cabling_[iter] );
  }
}


