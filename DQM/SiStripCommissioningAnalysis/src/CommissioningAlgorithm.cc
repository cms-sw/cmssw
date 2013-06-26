#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "TProfile.h"
#include <iomanip>

// ----------------------------------------------------------------------------
// 
CommissioningAlgorithm::CommissioningAlgorithm( CommissioningAnalysis* const anal )
  : anal_( anal )
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAlgorithm::CommissioningAlgorithm()
  : anal_(0)
{;}

// ----------------------------------------------------------------------------
// 
void CommissioningAlgorithm::analysis( const std::vector<TH1*>& histos ) { 
  if ( anal_ ) { anal()->reset(); }
  extract( histos );
  analyse();
}

// ----------------------------------------------------------------------------
// 
uint32_t CommissioningAlgorithm::extractFedKey( const TH1* const his ) {
  SiStripHistoTitle title( his->GetName() );
  return title.keyValue();
}
