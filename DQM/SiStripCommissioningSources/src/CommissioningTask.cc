#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"

#include <iostream>
#include <string> 

using namespace std;

// -----------------------------------------------------------------------------
//
CommissioningTask::CommissioningTask( DaqMonitorBEInterface* dqm,
				      const FedChannelConnection& conn,
				      const string& my_name ) :
  dqm_(dqm),
  updateFreq_(0),
  fillCntr_(0),
  connection_(conn),
  fedKey_(0),
  fecKey_(0),
  booked_(false),
  //fedId_(0),
  //fedCh_(0),
  myName_(my_name)
{
  LogDebug("Commissioning") << "[CommissioningTask::CommissioningTask]" 
			    << " Constructing object for FED id/ch " 
			    << connection_.fedId() << "/" 
			    << connection_.fedCh();
  fedKey_ = SiStripReadoutKey::key( connection_.fedId(), 
				    connection_.fedCh() );
  fecKey_ = SiStripControlKey::key( connection_.fecCrate(),
				    connection_.fecSlot(),
				    connection_.fecRing(),
				    connection_.ccuAddr(),
				    connection_.ccuChan(),
				    connection_.lldChannel() );
  
}

// -----------------------------------------------------------------------------
//
CommissioningTask::~CommissioningTask() {
  LogDebug("Commissioning") << "[CommissioningTask::CommissioningTask]" 
			    << " Destructing object for FED id/ch " 
			    << connection_.fedId() << "/" 
			    << connection_.fedCh();
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::book() {
  edm::LogError("Commissioning") << "[" << __PRETTY_FUNCTION__ << "]"
				 << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fill( const SiStripEventSummary& summary,
			      const edm::DetSet<SiStripRawDigi>& digis ) {
  edm::LogError("Commissioning") << "[" << __PRETTY_FUNCTION__ << "]"
				 << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fill( const SiStripEventSummary& summary,
			      const uint16_t& fed_id,
			      const map<uint16_t,float>& fed_ch ) {
  edm::LogError("Commissioning") << "[" << __PRETTY_FUNCTION__ << "]"
				 << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::update() {
  edm::LogError("Commissioning") << "[" << __PRETTY_FUNCTION__ << "]"
				 << " No derived implementation exists!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::bookHistograms() {
  edm::LogInfo("Commissioning") << "[CommissioningTask::book]"
				<< " Booking histograms for FED id/ch: "
				<< connection_.fedId() << "/"
				<< connection_.fedCh();
  book();
  booked_ = true;
  
  //   SiStripHistoNamingScheme::HistoTitle histo_title = SiStripHistoNamingScheme::histoTitle( title );
//   cout << "HistoTitle components: " 
//        << SiStripHistoNamingScheme::task( histo_title.task_ ) << " " 
//        << SiStripHistoNamingScheme::contents( histo_title.contents_ ) << " " 
//        << SiStripHistoNamingScheme::keyType( histo_title.keyType_ ) << " " 
//        << setfill('0') << setw(8) << hex << histo_title.keyValue_ << dec << " " 
//        << SiStripHistoNamingScheme::granularity( histo_title.granularity_ ) << " " 
//        << histo_title.channel_ << " " 
//        << histo_title.extraInfo_ << endl;

}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fillHistograms( const SiStripEventSummary& summary,
					const edm::DetSet<SiStripRawDigi>& digis ) {
  LogDebug("Commissioning") << "[CommissioningTask::fillHistograms]";
  if ( !booked_ ) {
    edm::LogError("Commissioning") << "[CommissioningTask::fillHistograms]"
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
					const map<uint16_t,float>& fed_ch ) {
  LogDebug("Commissioning") << "[CommissioningTask::fillHistograms]";
  if ( !booked_ ) {
    edm::LogError("Commissioning") << "[CommissioningTask::fillHistograms]"
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
  LogDebug("Commissioning") << "[CommissioningTask::updateHistograms]"
			    << " Updating histograms...";
  update();
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistoSet( HistoSet& histo_set, 
					const uint32_t& bin,
					const float& value ) {
  
  // Check bin number
  if ( bin >= histo_set.vNumOfEntries_.size() ) { 
    edm::LogError("Commissioning") << "[VpspScanTask::fill]" 
				   << "  Unexpected bin! " << bin;
    return;
  }
  
  // Set entries
  histo_set.vNumOfEntries_[bin]++;
  
  // Check if histo is TProfile or not
  if ( !histo_set.isProfile_ ) { return; }
  
  // Check bin number
  if ( bin >= histo_set.vSumOfContents_.size() || 
       bin >= histo_set.vSumOfSquares_.size() ) { 
    edm::LogError("Commissioning") << "[VpspScanTask::fill]" 
				   << "  Unexpected bin! " << bin;
    return;
  }
  
  // Set sum of contents and squares
  histo_set.vSumOfContents_[bin] += value;
  histo_set.vSumOfSquares_[bin] += value*value;
  
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::updateHistoSet( HistoSet& histo_set ) {
  
  // Check if histo exists
  if ( !histo_set.histo_ ) {
    edm::LogError("Commissioning") << "[CommissioningTask::updateHistoSet]" 
				   << " NULL pointer to ME!";
    return;
  }

  // Utility class that allows to update bin contents of TProfile histo
  static UpdateTProfile profile;
  
  // Extract TProfile object
  TProfile* prof = ExtractTObject<TProfile>().extract( histo_set.histo_ );
  // if ( prof ) { prof->SetErrorOption("s"); } //@@ necessary?
  
  // Update TProfile histo
  for ( uint32_t ibin = 0; ibin < histo_set.vNumOfEntries_.size(); ibin++ ) {
    if ( histo_set.isProfile_ ) {
      profile.setBinContents( prof,
			      ibin+1, 
			      histo_set.vNumOfEntries_[ibin],
			      histo_set.vSumOfContents_[ibin],
			      histo_set.vSumOfSquares_[ibin] );
    } else {
      histo_set.histo_->setBinContent( ibin+1, histo_set.vNumOfEntries_[ibin]*1. );
    }
  }
  
}


