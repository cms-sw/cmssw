#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "TProfile.h"

using namespace std;

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::analysis( const vector<TProfile*>& histos ) { 
  reset();
  extract( histos );
  analyse();
}

