#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "TProfile.h"

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const uint32_t& key,
					      const std::string& my_name ) 
  : fec_( SiStripFecKey(key) ),
    fed_(),
    myName_(my_name)
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const std::string& my_name ) 
  : fec_(),
    fed_(),
    myName_(my_name)
{;}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::analysis( const std::vector<TH1*>& histos ) { 
  reset();
  extract( histos );
  analyse();
}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::header( std::stringstream& ss ) const { 
  ss << myName() << " monitorables";
  /*if ( SiStripFecKey::key(fec()) ) { */ss << fec(); //}
  /*if ( SiStripFedKey::key(fed()) ) { */ss << fed(); //}
  ss << "\n";
}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::extractFedKey( const TH1* const his ) {
  SiStripHistoTitle title( his->GetName() );
  fed_ = SiStripFedKey( title.keyValue() );
}
