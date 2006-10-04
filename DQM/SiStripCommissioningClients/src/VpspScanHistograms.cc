#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::VpspScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui)
{
  cout << "[VpspScanHistograms::VpspScanHistograms]"
       << " Created object for PEDESTALS histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::~VpspScanHistograms() {
  cout << "[VpspScanHistograms::~VpspScanHistograms]" << endl;
}

