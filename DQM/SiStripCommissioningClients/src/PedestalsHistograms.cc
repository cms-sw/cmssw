#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::PedestalsHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui)
{
  cout << "[PedestalsHistograms::PedestalsHistograms]"
       << " Created object for pedestals histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::~PedestalsHistograms() {
  cout << "[PedestalsHistograms::~PedestalsHistograms]" << endl;
}
