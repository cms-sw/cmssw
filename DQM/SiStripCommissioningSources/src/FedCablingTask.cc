#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
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
  string info = "";
  for ( uint16_t iter = 0; iter < 2; iter++ ) {
    
    // Define number of histo bins and title
    if ( iter == 0 )      { nbins = 1024; info = sistrip::fedId_; }
    else if ( iter == 1 ) { nbins = 96;   info = sistrip::fedChannel_; }
    else {
      edm::LogError("Commissioning") << "[FedCablingTask::book]"
				     << " Unexpected number of HistoSets" << iter;
    }
    
    title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::FED_CABLING,
						  SiStripHistoNamingScheme::SUM2, 
						  SiStripHistoNamingScheme::FED, 
						  fedKey(),
						  SiStripHistoNamingScheme::LLD_CHAN, 
						  connection().lldChannel(),
						  info );
    cabling_[iter].meSumOfSquares_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
    
    title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::FED_CABLING,
						  SiStripHistoNamingScheme::SUM, 
						  SiStripHistoNamingScheme::FED, 
						  fedKey(),
						  SiStripHistoNamingScheme::LLD_CHAN, 
						  connection().lldChannel(),
						  info );
    cabling_[iter].meSumOfContents_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
    
    title = SiStripHistoNamingScheme::histoTitle( SiStripHistoNamingScheme::FED_CABLING,
						  SiStripHistoNamingScheme::NUM, 
						  SiStripHistoNamingScheme::FED, 
						  fedKey(),
						  SiStripHistoNamingScheme::LLD_CHAN, 
						  connection().lldChannel(),
						  info );
    cabling_[iter].meNumOfEntries_ = dqm()->book1D( title, title, nbins, -0.5, nbins*1.-0.5 );
    
    cabling_[iter].vSumOfSquares_.resize(nbins,0);
    cabling_[iter].vSumOfSquaresOverflow_.resize(nbins,0);
    cabling_[iter].vSumOfContents_.resize(nbins,0);
    cabling_[iter].vNumOfEntries_.resize(nbins,0);
    
  }
  
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::fill( const SiStripEventSummary& summary,
			   const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[FedCablingTask::fill]";
  
  stringstream ss; 
  ss << "[FedCablingTask::fill] DeviceId: " 
     << setfill('0') << setw(8) << hex << summary.deviceId() << dec;
  LogDebug("Commissioning") << ss.str();
  
  //@@ if scope mode length is in trigger fed, then 
  //@@ can add check here on number of digis
  if ( digis.data.empty() ) {
    edm::LogError("Commissioning") << "[FedCablingTask::fill]" 
				   << " Unexpected number of digis! " 
				   << digis.data.size(); 
  } else {
    // Determine ADC median level
    vector<uint16_t> level;
    level.reserve(128); 
    for ( uint16_t idigi = 0; idigi < digis.data.size(); idigi++ ) { level.push_back( digis.data[idigi].adc() ); }
    sort( level.begin(), level.end() ); 
    uint16_t index = level.size()%2 ? level.size()/2 : level.size()/2-1;
    if ( !level.empty() ) {
      // Fill FED id histo
      if ( fedId() < cabling_[0].vNumOfEntries_.size() ) { 
	updateHistoSet( cabling_[0], fedId(), level[index] );
      } else {
	edm::LogError("Commissioning") << "[FedCablingTask::fill]" 
				       << "  Unexpected FED id! " << fedId();
	return;
      }
      // Fill FED channel histo
      if ( fedCh() < cabling_[1].vNumOfEntries_.size() ) { 
	updateHistoSet( cabling_[1], fedCh(), level[index] );
      } else {
	edm::LogError("Commissioning") << "[FedCablingTask::fill]" 
				       << "  Unexpected FED channel! " << fedCh();
	return;
      }
    }
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


