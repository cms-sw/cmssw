#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <iostream>
#include <string> 

using namespace std;

// -----------------------------------------------------------------------------
//
CommissioningTask::CommissioningTask( DaqMonitorBEInterface* dqm,
				      const SiStripModule& module ) :
  dqm_(dqm),
  updateFreq_(0),
  fillCntr_(0)
{
  cout << "[CommissioningTask::CommissioningTask]" 
       << " Constructing object..." << endl;
  book( module );
}

// -----------------------------------------------------------------------------
//
CommissioningTask::~CommissioningTask() {
  cout << "[CommissioningTask::~CommissioningTask]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::book( const SiStripModule& ) {
  cerr << "[CommissioningTask::book]"
       << " This virtual method should always be over-ridden!" << endl;
}

// -----------------------------------------------------------------------------
//
void CommissioningTask::fillHistograms( const vector<StripDigi>& digis ) {
  fillCntr_++;
  fill( digis ); 
  if ( updateFreq_ ) { if ( !(fillCntr_%updateFreq_) ) update(); }
}





