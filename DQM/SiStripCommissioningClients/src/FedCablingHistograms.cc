#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::FedCablingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui)
{
  cout << "[FedCablingHistograms::FedCablingHistograms]"
       << " Created object for FED CABLING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::~FedCablingHistograms() {
  cout << "[FedCablingHistograms::~FedCablingHistograms]" << endl;
}
