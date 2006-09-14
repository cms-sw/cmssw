#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::OptoScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui)
{
  cout << "[OptoScanHistograms::OptoScanHistograms]"
       << " Created object for OPTO (bias and gain) SCAN histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::~OptoScanHistograms() {
  cout << "[OptoScanHistograms::~OptoScanHistograms]" << endl;
}
