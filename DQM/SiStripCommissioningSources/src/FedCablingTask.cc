#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
FedCablingTask::FedCablingTask( DQMStore* dqm,
				const FedChannelConnection& conn ) :
  CommissioningTask( dqm, conn, "FedCablingTask" ),
  histos_()
{}

// -----------------------------------------------------------------------------
//
FedCablingTask::~FedCablingTask() {
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::book() {
  
  histos_.resize(2);
  
  std::string title;
  uint16_t nbins = 0;
  std::string extra_info = "";
  for ( uint16_t iter = 0; iter < 2; iter++ ) {
      
    // Define number of histo bins and title
    if ( iter == 0 )      { nbins = 1024; extra_info = sistrip::feDriver_; }
    else if ( iter == 1 ) { nbins = 96;   extra_info = sistrip::fedChannel_; }
    else {
      edm::LogWarning(mlDqmSource_)
	<< "[FedCablingTask::" << __func__ << "]"
	<< " Unexpected number of HistoSets: " << iter;
    }
      
    title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
			       sistrip::FED_CABLING,
			       sistrip::FED_KEY, 
			       fedKey(),
			       sistrip::LLD_CHAN, 
			       connection().lldChannel(),
			       extra_info ).title();
      
    histos_[iter].histo( dqm()->bookProfile( title, title, 
					     nbins, -0.5, nbins*1.-0.5,
					     1025, 0., 1025. ) );
      
    histos_[iter].vNumOfEntries_.resize(nbins,0);
    histos_[iter].vSumOfContents_.resize(nbins,0);
    histos_[iter].vSumOfSquares_.resize(nbins,0);

  }

}

// -----------------------------------------------------------------------------
//
void FedCablingTask::fill( const SiStripEventSummary& summary,
			   const uint16_t& fed_id,
			   const std::map<uint16_t,float>& fed_ch ) {
  
  if ( fed_ch.empty() ) { 
    edm::LogWarning(mlDqmSource_)  
      << "[FedCablingTask::" << __func__ << "]"
      << " No FED channels with high signal!";
    return; 
  } else {
    LogTrace(mlDqmSource_)  
      << "[FedCablingTask::" << __func__ << "]"
      << " Found " << fed_ch.size()
      << " FED channels with high signal!";
  }
  
  std::map<uint16_t,float>::const_iterator ichan = fed_ch.begin();
  for ( ; ichan != fed_ch.end(); ichan++ ) {
    updateHistoSet( histos_[0], fed_id, ichan->second );
    updateHistoSet( histos_[1], ichan->first, ichan->second );
  } 
  
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::update() {
  for ( uint32_t iter = 0; iter < histos_.size(); iter++ ) {
    updateHistoSet( histos_[iter] );
  }
}



