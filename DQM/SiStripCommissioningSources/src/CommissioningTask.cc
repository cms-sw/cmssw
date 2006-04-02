#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <string> 

using namespace std;

// -----------------------------------------------------------------------------
//
CommissioningTask::CommissioningTask( DaqMonitorBEInterface* dqm,
				      const FedChannelConnection& conn ) :
  dqm_(dqm),
  updateFreq_(0),
  fillCntr_(0),
  connection_(conn),
  booked_(false)
{
  LogDebug("Commissioning") << "[CommissioningTask::CommissioningTask]" 
			    << " Constructing object for FED id/ch " 
			    << connection_.fedId() << "/" 
			    << connection_.fedCh();
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
void CommissioningTask::book( const FedChannelConnection& ) {
  edm::LogError("Commissioning") << "[CommissioningTask::book]"
				 << " This virtual method should always be over-ridden!";
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::bookHistograms() {
  edm::LogInfo("Commissioning") << "[CommissioningTask::book]"
				<< " Booking histograms for FED id/ch: "
				<< connection_.fedId() << "/"
				<< connection_.fedCh();
  book( connection_ );
  booked_ = true;
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
  if ( updateFreq_ ) { if ( !(fillCntr_%updateFreq_) ) update(); }
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
string CommissioningTask::title( string variable, string contents, uint32_t lld_channel ) {
  static string sep("|");
  static stringstream ss; ss.str(""); 
  if ( contents != "" ) { ss << variable << sep << contents << sep << "LLDchan" << lld_channel; }
  else { ss << variable << sep << lld_channel; }
  return ss.str();
}
