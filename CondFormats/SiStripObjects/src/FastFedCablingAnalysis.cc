#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
const float FastFedCablingAnalysis::threshold_ = 100.; // [ADC]

// ----------------------------------------------------------------------------
// 
const float FastFedCablingAnalysis::dirtyThreshold_ = 800; // [ADC]

// ----------------------------------------------------------------------------
// 
const float FastFedCablingAnalysis::trimDacThreshold_ = 10; // [ADC]

// ----------------------------------------------------------------------------
// 
const uint16_t FastFedCablingAnalysis::nBitsForDcuId_ = 32;

// ----------------------------------------------------------------------------
// 
const uint16_t FastFedCablingAnalysis::nBitsForLldCh_ = 2;

// ----------------------------------------------------------------------------
// 
FastFedCablingAnalysis::FastFedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::fastCablingAnalysis_),
    dcuHardId_(sistrip::invalid32_), 
    lldCh_(sistrip::invalid_),
    highMedian_(1.*sistrip::invalid_),
    highMean_(1.*sistrip::invalid_),
    highRms_(1.*sistrip::invalid_),
    lowMedian_(1.*sistrip::invalid_),
    lowMean_(1.*sistrip::invalid_),
    lowRms_(1.*sistrip::invalid_),
    range_(1.*sistrip::invalid_),
    midRange_(1.*sistrip::invalid_),
    max_(1.*sistrip::invalid_),
    min_(1.*sistrip::invalid_)
{
  fecKey( SiStripFecKey( sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_ ).key() );
  fedKey( key );
}

// ----------------------------------------------------------------------------
// 
FastFedCablingAnalysis::FastFedCablingAnalysis() 
  : CommissioningAnalysis(sistrip::fastCablingAnalysis_),
    dcuHardId_(sistrip::invalid32_), 
    lldCh_(sistrip::invalid_),
    highMedian_(1.*sistrip::invalid_),
    highMean_(1.*sistrip::invalid_),
    highRms_(1.*sistrip::invalid_),
    lowMedian_(1.*sistrip::invalid_),
    lowMean_(1.*sistrip::invalid_),
    lowRms_(1.*sistrip::invalid_),
    range_(1.*sistrip::invalid_),
    midRange_(1.*sistrip::invalid_),
    max_(1.*sistrip::invalid_),
    min_(1.*sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::reset() {
    dcuHardId_ = sistrip::invalid32_; 
    lldCh_ = sistrip::invalid_;
    highMedian_ = 1.*sistrip::invalid_;
    highMean_ = 1.*sistrip::invalid_;
    highRms_ = 1.*sistrip::invalid_;
    lowMedian_ = 1.*sistrip::invalid_;
    lowMean_ = 1.*sistrip::invalid_;
    lowRms_ = 1.*sistrip::invalid_;
    range_ = 1.*sistrip::invalid_;
    midRange_ = 1.*sistrip::invalid_;
    max_ = 1.*sistrip::invalid_;
    min_ = 1.*sistrip::invalid_; 
}

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::isValid() const {
  return ( dcuHardId_ < sistrip::invalid32_ &&
	   lldCh_ < sistrip::valid_ && 
	   highMedian_ < sistrip::valid_ && 
	   highMean_ < sistrip::valid_ && 
	   highRms_ < sistrip::valid_ && 
	   lowMedian_ < sistrip::valid_ && 
	   lowMean_ < sistrip::valid_ && 
	   lowRms_ < sistrip::valid_ && 
	   range_ < sistrip::valid_ && 
	   midRange_ < sistrip::valid_ && 
	   max_ < sistrip::valid_ && 
	   min_ < sistrip::valid_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::isDirty() const {
  return ( highMean_ < dirtyThreshold_ );
} 

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::badTrimDac() const {
  return ( lowMean_ < trimDacThreshold_ );
} 

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::header( std::stringstream& ss ) const { 
  ss << "[" << myName() << "] Monitorables (65535 means \"invalid\"):" << std::endl;

  //summary(ss);

  SiStripFecKey fec_key( fecKey() );
  if ( fec_key.isValid() ) {
    ss << " Crate/FEC/Ring/CCU/Mod/LLD     : " 
       << fec_key.fecCrate() << "/" 
       << fec_key.fecSlot() << "/" 
       << fec_key.fecRing() << "/" 
       << fec_key.ccuAddr() << "/" 
       << fec_key.ccuChan() << "/" 
       << fec_key.lldChan() 
       << std::endl;
  } else {
    ss << " Crate/FEC/Ring/CCU/Mod/LLD     : (invalid)" 
       << std::endl;
  }

  SiStripFedKey fed_key( fedKey() );
  ss << " FedId/FeUnit/FeChan/FedChannel : " 
     << fed_key.fedId() << "/" 
     << fed_key.feUnit() << "/" 
     << fed_key.feChan() << "/"
     << fed_key.fedChannel()
     << std::endl;
  // if ( fed_key.fedChannel() != sistrip::invalid_ ) { ss << fed_key.fedChannel(); }
  // else { ss << "(invalid)"; }
  // ss << std::endl;
  
  ss << " FecKey/Fedkey (hex)            : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fecKey()
     << " / 0x" 
     << std::setw(8) << std::setfill('0') << fedKey()
     << std::dec
     << std::endl;
  
  ss << " DcuId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << dcuId() 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << dcuId() 
     << std::endl;

  ss << " DetId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << detId() 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << detId() 
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 type,
					 sistrip::FED_KEY, 
					 fed_key.key(),
					 sistrip::LLD_CHAN, 
					 fec_key.lldChan() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::readoutView_ << ":"
     << fed_key.fedId() << "/" 
     << fed_key.feUnit() << "/" 
     << fed_key.feChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << fed_key.path()
     << ":"
     << title
     << std::endl;

}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss <<  std::fixed << std::setprecision(2)
     << " DCU id extracted from histo     : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << dcuHardId_ << std::endl
     << std::dec
     << " LLD chan extracted from histo   : " <<  ( lldCh_ > 3 ? sistrip::invalid_ : lldCh_ )  << std::endl
     << " \"High\" level (mean+/-rms) [ADC] : " << highMean_ << " +/- " << highRms_ << std::endl
     << " \"Low\" level (mean+/-rms)  [ADC] : " << lowMean_ << " +/- " << lowRms_ << std::endl
     << " Median \"high\" level       [ADC] : " << highMedian_ << std::endl
     << " Median \"low\" level        [ADC] : " << lowMedian_ << std::endl
     << " Range                     [ADC] : " << range_ << std::endl
     << " Mid-range level           [ADC] : " << midRange_ << std::endl
     << " Maximum level             [ADC] : " << max_ << std::endl
     << " Minimum level             [ADC] : " << min_ << std::endl;
  ss << std::boolalpha
     << " isValid                         : " << isValid()  << std::endl
     << " isDirty                         : " << isDirty()  << std::endl
     << " badTrimDac                      : " << badTrimDac()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ")          : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}
