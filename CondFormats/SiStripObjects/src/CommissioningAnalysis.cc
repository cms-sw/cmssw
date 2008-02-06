#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "TProfile.h"
#include <iomanip>

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const uint32_t& key,
					      const std::string& my_name ) 
  : fecKey_( SiStripFecKey(key) ),
    fedKey_(),
    dcuId_(sistrip::invalid32_),
    detId_(sistrip::invalid32_),
    myName_(my_name),
    errors_(VString(0,""))
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const std::string& my_name ) 
  : fecKey_(),
    fedKey_(),
    dcuId_(sistrip::invalid32_),
    detId_(sistrip::invalid32_),
    myName_(my_name),
    errors_(VString(0,""))
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

  //summary(ss);
  
  ss << " Crate/FEC/Ring/CCU/Mod/LLD     : " 
     << fecKey().fecCrate() << "/" 
     << fecKey().fecSlot() << "/" 
     << fecKey().fecRing() << "/" 
     << fecKey().ccuAddr() << "/" 
     << fecKey().ccuChan() << "/" 
     << fecKey().lldChan() 
     << std::endl;

  ss << " FedId/FeUnit/FeChan/FedChannel : " 
     << fedKey().fedId() << "/" 
     << fedKey().feUnit() << "/" 
     << fedKey().feChan() << "/"
     << fedKey().fedChannel()
     << std::endl;
  // if ( fedKey().fedChannel() != sistrip::invalid_ ) { ss << fedKey().fedChannel(); }
  // else { ss << "(invalid)"; }
  // ss << std::endl;
  
  ss << " FecKey/Fedkey (hex)            : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fecKey().key()
     << " / 0x" 
     << std::setw(8) << std::setfill('0') << fedKey().key() 
     << std::dec
     << std::endl;
  
  ss << " DcuId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << dcuId_ 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << dcuId_ 
     << std::endl;

  ss << " DetId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << detId_ 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << detId_ 
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::summary( std::stringstream& ss ) const { 
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 type,
					 sistrip::FED_KEY, 
					 fedKey().key(),
					 sistrip::LLD_CHAN, 
					 fecKey().lldChan() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::controlView_ << ":"
     << fecKey().fecCrate() << "/" 
     << fecKey().fecSlot() << "/" 
     << fecKey().fecRing() << "/" 
     << fecKey().ccuAddr() << "/" 
     << fecKey().ccuChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << SiStripFecKey( fecKey().fecCrate(),
		       fecKey().fecSlot(), 
		       fecKey().fecRing(), 
		       fecKey().ccuAddr(), 
		       fecKey().ccuChan() ).path()
     << ":"
     << title
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::extractFedKey( const TH1* const his ) {
  SiStripHistoTitle title( his->GetName() );
  fedKey_ = SiStripFedKey( title.keyValue() );
}
