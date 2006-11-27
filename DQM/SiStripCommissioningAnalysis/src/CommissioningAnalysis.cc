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
void CommissioningAnalysis::analysis( const vector<TH1*>& histos ) { 
  reset();
  extract( histos );
  analyse();
}

