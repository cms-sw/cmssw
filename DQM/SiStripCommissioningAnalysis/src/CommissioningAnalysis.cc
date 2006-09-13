#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "TProfile.h"

using namespace std;

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const uint32_t& key ) 
  : key_(key) 
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis() 
  : key_(0) 
{;}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::analysis( const vector<TProfile*>& histos ) { 
  reset();
  extract( histos );
  analyse();
}

