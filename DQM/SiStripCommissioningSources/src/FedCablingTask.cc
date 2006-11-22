#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
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
{}

// -----------------------------------------------------------------------------
//
FedCablingTask::~FedCablingTask() {
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
      edm::LogWarning(mlDqmSource_)
	<< "[FedCablingTask::" << __func__ << "]"
	<< " Unexpected number of HistoSets: " << iter;
    }
    
    title = SiStripHistoNamingScheme::histoTitle( HistoTitle( sistrip::FED_CABLING,
							      sistrip::FED_KEY, 
							      fedKey(),
							      sistrip::LLD_CHAN, 
							      connection().lldChannel(),
							      extra_info ) );

    cabling_[iter].histo_ = dqm()->bookProfile( title, title, 
						nbins, -0.5, nbins*1.-0.5,
						1025, 0., 1025. );
    
    cabling_[iter].vNumOfEntries_.resize(nbins,0);
    cabling_[iter].vSumOfContents_.resize(nbins,0);
    cabling_[iter].vSumOfSquares_.resize(nbins,0);
    //cabling_[iter].isProfile_ = false;
    
  }
  
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::fill( const SiStripEventSummary& summary,
			   const uint16_t& fed_id,
			   const map<uint16_t,float>& fed_ch ) {

  if ( fed_ch.empty() ) { 
    edm::LogWarning(mlDqmSource_)  
      << "[FedCablingTask::" << __func__ << "]"
      << " No FED channels with high signal!";
    return; 
  }
  
  map<uint16_t,float>::const_iterator ichan = fed_ch.begin();
  for ( ; ichan != fed_ch.end(); ichan++ ) {
    updateHistoSet( cabling_[0], fed_id, ichan->second );
    updateHistoSet( cabling_[1], ichan->first, ichan->second );
  } 
  
}

// -----------------------------------------------------------------------------
//
void FedCablingTask::update() {
  for ( uint32_t iter = 0; iter < cabling_.size(); iter++ ) {
    updateHistoSet( cabling_[iter] );
  }
}


