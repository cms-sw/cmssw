#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <iomanip>

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const uint32_t& key,
					      const std::string& my_name ) 
  : fecKey_( key ),
    fedKey_(sistrip::invalid32_),
    dcuId_(sistrip::invalid32_),
    detId_(sistrip::invalid32_),
    myName_(my_name),
    errors_(VString(0,""))
{;}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::CommissioningAnalysis( const std::string& my_name ) 
  : fecKey_(sistrip::invalid32_),
    fedKey_(sistrip::invalid32_),
    dcuId_(sistrip::invalid32_),
    detId_(sistrip::invalid32_),
    myName_(my_name),
    errors_(VString(0,""))
{;}

// ----------------------------------------------------------------------------
// 
void CommissioningAnalysis::header( std::stringstream& ss ) const { 
  ss << "[" << myName() << "] Monitorables (65535 means \"invalid\"):" << std::endl;

  //summary(ss);

  SiStripFecKey fec_key( fecKey_ );
  ss << " Crate/FEC/Ring/CCU/Mod/LLD     : " 
     << fec_key.fecCrate() << "/" 
     << fec_key.fecSlot() << "/" 
     << fec_key.fecRing() << "/" 
     << fec_key.ccuAddr() << "/" 
     << fec_key.ccuChan() << "/" 
     << fec_key.lldChan() 
     << std::endl;

  SiStripFedKey fed_key( fedKey_ );
  ss << " FedId/FeUnit/FeChan/FedChannel : " 
     << fed_key.fedId() << "/" 
     << fed_key.feUnit() << "/" 
     << fed_key.feChan() << "/"
     << fed_key.fedChannel()
     << std::endl;
  // if ( fed_key.fedChannel() != sistrip::invalid_ ) { ss << fed_key.fedChannel(); }
  // else { ss << "(invalid)"; }
  // ss << std::endl;
  
//   ss << " FecKey/Fedkey (hex)            : 0x" 
//      << std::hex 
//      << std::setw(8) << std::setfill('0') << fecKey_
//      << " / 0x" 
//      << std::setw(8) << std::setfill('0') << fedKey_
//      << std::dec
//      << std::endl;
  
  ss << " FecKey (hex/dec)               : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fecKey_ 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << fecKey_ 
     << std::endl;

  ss << " FedKey (hex/dec)               : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fedKey_ 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << fedKey_ 
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

  SiStripFecKey fec_key( fecKey_ );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 type,
					 sistrip::FED_KEY, 
					 fedKey(),
					 sistrip::LLD_CHAN, 
					 fec_key.lldChan() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::controlView_ << ":"
     << fec_key.fecCrate() << "/" 
     << fec_key.fecSlot() << "/" 
     << fec_key.fecRing() << "/" 
     << fec_key.ccuAddr() << "/" 
     << fec_key.ccuChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << SiStripFecKey( fec_key.fecCrate(),
		       fec_key.fecSlot(), 
		       fec_key.fecRing(), 
		       fec_key.ccuAddr(), 
		       fec_key.ccuChan() ).path()
     << ":"
     << title
     << std::endl;
  
}
