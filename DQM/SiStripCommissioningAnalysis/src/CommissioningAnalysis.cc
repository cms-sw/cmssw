#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "TProfile.h"
#include <iomanip>

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const uint32_t& key,
					      const std::string& my_name ) 
  : fec_( SiStripFecKey(key) ),
    fed_(),
    myName_(my_name),
    errors_(VStrings(0,""))
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const std::string& my_name ) 
  : fec_(),
    fed_(),
    myName_(my_name),
    errors_(VStrings(0,""))
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
  ss << "[" << myName() << "] Monitorables:" << std::endl;
  ss << " FecKey/FedKey               : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fec().key() << "/0x" 
     << std::setw(8) << std::setfill('0') << fed().key() << std::endl
     << std::dec
     << " Crate/FEC/Ring/CCU/Mod/LLD  : " 
     << fec().fecCrate() << "/" 
     << fec().fecSlot() << "/" 
     << fec().fecRing() << "/" 
     << fec().ccuAddr() << "/" 
     << fec().ccuChan() << "/" 
     << fec().lldChan() << std::endl
     << " FedId/FeUnit/FeChan/FedChan : " 
     << fed().fedId() << "/" 
     << fed().feUnit() << "/" 
     << fed().feChan() << "/";
  if ( fed().fedChannel() != sistrip::invalid_ ) {
    ss << fed().fedChannel() << std::endl;
  } else { ss << "(invalid)" << std::endl; }
}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::extractFedKey( const TH1* const his ) {
  SiStripHistoTitle title( his->GetName() );
  fed_ = SiStripFedKey( title.keyValue() );
}
