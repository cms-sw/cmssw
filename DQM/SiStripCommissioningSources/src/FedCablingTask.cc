#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
FedCablingTask::FedCablingTask( DaqMonitorBEInterface* dqm,
			    const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "FedCablingTask" ),
  cabling_()
{
  LogDebug(mlCommSource_)
    << "[FedCablingTask::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
FedCablingTask::~FedCablingTask() {
  LogDebug(mlCommSource_)
    << "[FedCablingTask::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::book() {
  
  cabling_.resize(2);
  
  string title;
  uint16_t nbins = 0;
  string extra_info = "";
  for ( uint16_t iter = 0; iter < 2; iter++ ) {
    
    // Define number of histo bins and title
    if ( iter == 0 )      { nbins = 1024; extra_info = sistrip::fedId_; }
    else if ( iter == 1 ) { nbins = 96;   extra_info = sistrip::fedChannel_; }
    else {
      edm::LogWarning(mlCommSource_)
	<< "[FedCablingTask::" << __func__ << "]"
	<< " Unexpected number of HistoSets: " << iter;
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

// -----------------------------------------------------------------------------
//
void FedCablingTask::fill( const SiStripEventSummary& summary,
			   const uint16_t& fed_id,
			   const map<uint16_t,float>& fed_ch ) {
  
  if ( fed_ch.empty() ) { 
    edm::LogWarning(mlCommSource_) 
      << "[FedCablingTask::" << __func__ << "]"
      << " No FED channels with high signal!";
    return; 
  }
  
  // Fill FED id and channel histogram
  map<uint16_t,float>::const_iterator ichan = fed_ch.begin();
  for ( ; ichan != fed_ch.end(); ichan++ ) {
    updateHistoSet( cabling_[0], fed_id, ichan->second );
    updateHistoSet( cabling_[1], ichan->first, ichan->second );
    LogDebug(mlCommSource_) 
      << "[FedCablingTask::" << __func__ << "]"
      << " Found possible connection between device "
      << setfill('0') << setw(8) << hex << summary.deviceId() << dec
      << " with control path " 
      << connection().fecCrate() << "/"
      << connection().fecSlot() << "/"
      << connection().fecRing() << "/"
      << connection().ccuAddr() << "/"
      << connection().ccuChan() << "/"
      << connection().lldChannel()
      << " and FED id/channel "
      << fed_id << "/" << ichan->first
      << " with signal " << ichan->second 
      << " [adc] over background " << "XXX +/- YYY [adc]" 
      << "(S/N = " << "ZZZ" << ")";

  } 
  
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::update() {
  for ( uint32_t iter = 0; iter < cabling_.size(); iter++ ) {
    updateHistoSet( cabling_[iter] );
  }
}


