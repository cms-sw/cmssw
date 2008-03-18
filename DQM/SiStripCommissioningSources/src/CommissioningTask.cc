#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
CommissioningTask::CommissioningTask( DQMStore* dqm,
				      const FedChannelConnection& conn,
				      const std::string& my_name ) :
  dqm_(dqm),
  updateFreq_(0),
  fillCntr_(0),
  connection_(conn),
  fedKey_(0),
  fecKey_(0),
  booked_(false),
  myName_(my_name),
  eventSetup_(0)
{
  uint16_t fed_ch = connection_.fedCh();
  fedKey_ = SiStripFedKey( connection_.fedId(), 
			   SiStripFedKey::feUnit(fed_ch),
			   SiStripFedKey::feChan(fed_ch) ).key();
  fecKey_ = SiStripFecKey( connection_.fecCrate(),
			   connection_.fecSlot(),
			   connection_.fecRing(),
			   connection_.ccuAddr(),
			   connection_.ccuChan(),
			   connection_.lldChannel() ).key();
  
  LogTrace(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]" 
    << " Constructing '" << myName_
    << "' object for FecKey/FedKey: "
    << "0x" << std::hex << std::setw(8) << std::setfill('0') << fecKey_ << std::dec
    << "/"
    << "0x" << std::hex << std::setw(8) << std::setfill('0') << fedKey_ << std::dec
    << " and Crate/FEC/ring/CCU/module/LLDchan: " 
    << connection_.fecCrate() << "/"
    << connection_.fecSlot() << "/" 
    << connection_.fecRing() << "/" 
    << connection_.ccuAddr() << "/" 
    << connection_.ccuChan() << "/" 
    << connection_.lldChannel() 
    << " and FedId/Ch: " 
    << connection_.fedId() << "/" 
    << connection_.fedCh();
}

// -----------------------------------------------------------------------------
//
CommissioningTask::~CommissioningTask() {
  LogTrace(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]" 
    << " Destructing object for FED id/ch " 
    << " Constructing '" << myName_
    << "' object for FecKey/FedKey: "
    << "0x" << std::hex << std::setw(8) << std::setfill('0') << fecKey_ << std::dec
    << "/"
    << "0x" << std::hex << std::setw(8) << std::setfill('0') << fedKey_ << std::dec
    << " and Crate/FEC/ring/CCU/module/LLDchan: " 
    << connection_.fecCrate() << "/"
    << connection_.fecSlot() << "/" 
    << connection_.fecRing() << "/" 
    << connection_.ccuAddr() << "/" 
    << connection_.ccuChan() << "/" 
    << connection_.lldChannel() 
    << " and FedId/Ch: " 
    << connection_.fedId() << "/" 
    << connection_.fedCh();
  //@@ do not delete EventSetup pointer!
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::book() {
  edm::LogWarning(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]"
    << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fill( const SiStripEventSummary& summary,
			      const edm::DetSet<SiStripRawDigi>& digis ) {
  edm::LogWarning(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]"
    << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fill( const SiStripEventSummary& summary,
			      const uint16_t& fed_id,
			      const std::map<uint16_t,float>& fed_ch ) {
  edm::LogWarning(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]"
    << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::update() {
  edm::LogWarning(mlDqmSource_)
    << "[CommissioningTask::" << __func__ << "]"
    << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::bookHistograms() {
  book();
  booked_ = true;
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fillHistograms( const SiStripEventSummary& summary,
					const edm::DetSet<SiStripRawDigi>& digis ) {
  if ( !booked_ ) {
    edm::LogWarning(mlDqmSource_)
      << "[CommissioningTask::" << __func__ << "]"
      << " Attempting to fill histos that haven't been booked yet!";
    return;
  }
  fillCntr_++;
  fill( summary, digis ); 
  if ( updateFreq_ && !(fillCntr_%updateFreq_) ) { 
    update(); 
  }
  
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fillHistograms( const SiStripEventSummary& summary,
					const uint16_t& fed_id,
					const std::map<uint16_t,float>& fed_ch ) {
  if ( !booked_ ) {
    edm::LogWarning(mlDqmSource_)
      << "[CommissioningTask::" << __func__ << "]"
      << " Attempting to fill histos that haven't been booked yet!";
    return;
  }
  fillCntr_++;
  fill( summary, fed_id, fed_ch ); 
  if ( updateFreq_ && !(fillCntr_%updateFreq_) ) { 
    update(); 
  }
  
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistograms() {
  update();
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistoSet( HistoSet& histo_set, 
					const uint32_t& bin ) {
  float value = 1.;
  updateHistoSet( histo_set, bin, value );
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistoSet( HistoSet& histo_set, 
					const uint32_t& bin,
					const float& value ) {
  
  // Check bin number
  if ( bin >= histo_set.vNumOfEntries_.size() ) { 
    edm::LogWarning(mlDqmSource_)
      << "[CommissioningTask::" << __func__ << "]"
      << " Unexpected bin number " << bin 
      << " when filling histogram of size " << histo_set.vNumOfEntries_.size();
    return;
  }
  
  // Check if histo is TProfile or not
  if ( !histo_set.isProfile_ ) {
    // Set entries
    histo_set.vNumOfEntries_[bin]+=value;
  } else {
    // Set entries
    histo_set.vNumOfEntries_[bin]++;
    
    // Check bin number
    if ( bin >= histo_set.vSumOfContents_.size() || 
         bin >= histo_set.vSumOfSquares_.size() ) { 
      edm::LogWarning(mlDqmSource_)
        << "[CommissioningTask::" << __func__ << "]"
        << " Unexpected bin when filling histogram: " << bin;
      return;
    }
    
    // Set sum of contents and squares
    histo_set.vSumOfContents_[bin] += value;
    histo_set.vSumOfSquares_[bin] += value*value;
  }

}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistoSet( HistoSet& histo_set ) {
  
  // Check if histo exists
  if ( !histo_set.histo_ ) {
    edm::LogWarning(mlDqmSource_)
      << "[CommissioningTask::" << __func__ << "]"
      << " NULL pointer to MonitorElement!";
    return;
  }

  if ( histo_set.isProfile_ ) {

    TProfile* prof = ExtractTObject<TProfile>().extract( histo_set.histo_ );
    // if ( prof ) { prof->SetErrorOption("s"); } //@@ necessary?
    static UpdateTProfile profile;
    for ( uint32_t ibin = 0; ibin < histo_set.vNumOfEntries_.size(); ibin++ ) {
      profile.setBinContents( prof,
			      ibin+1, 
			      histo_set.vNumOfEntries_[ibin],
			      histo_set.vSumOfContents_[ibin],
			      histo_set.vSumOfSquares_[ibin] );
    }

  } else {

    for ( uint32_t ibin = 0; ibin < histo_set.vNumOfEntries_.size(); ibin++ ) {
      histo_set.histo_->setBinContent( ibin+1, histo_set.vNumOfEntries_[ibin] );
    }
    
  }
  
}


