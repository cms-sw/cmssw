#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"

using namespace std;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui ) 
  : mui_(mui)
{
  cout << "[CommissioningHistograms::CommissioningHistograms]" 
       << " Created base object!" << endl;
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << "[CommissioningHistograms::~CommissioningHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHistos() {
  cout << "[CommissioningHistograms::createSummaryHistos]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createTrackerMap() {
  cout << "[CommissioningHistograms::createTrackerMap]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << "[CommissioningHistograms::uploadToConfigDb]" 
       << " (Derived) implementation to come..." << endl;
}



