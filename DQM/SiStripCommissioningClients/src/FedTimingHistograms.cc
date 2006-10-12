#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::FedTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui)
{
  cout << "[FedTimingHistograms::FedTimingHistograms]"
       << " Created object for FED TIMING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::~FedTimingHistograms() {
  cout << "[FedTimingHistograms::~FedTimingHistograms]" << endl;
}
