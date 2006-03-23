#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>
#include <string> 

using namespace std;

// -----------------------------------------------------------------------------
//
CommissioningTask::CommissioningTask( DaqMonitorBEInterface* dqm,
				      const FedChannelConnection& conn ) :
  dqm_(dqm),
  updateFreq_(0),
  fillCntr_(0)
{
  cout << "[CommissioningTask::CommissioningTask]" 
       << " Constructing object..." << endl;
  book( conn );
}

// -----------------------------------------------------------------------------
//
CommissioningTask::~CommissioningTask() {
  cout << "[CommissioningTask::~CommissioningTask]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::book( const FedChannelConnection& ) {
  cerr << "[CommissioningTask::book]"
       << " This virtual method should always be over-ridden!" << endl;
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fillHistograms( const SiStripEventSummary& summary,
					const edm::DetSet<SiStripRawDigi>& digis ) {
  fillCntr_++;
  fill( summary, digis ); 
  if ( updateFreq_ ) { if ( !(fillCntr_%updateFreq_) ) update(); }
}





