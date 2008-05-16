#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
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
const uint16_t OptoScanAnalysis::defaultGainSetting_ = sistrip::invalid_; //@@ 

// ----------------------------------------------------------------------------
// 
const uint16_t OptoScanAnalysis::defaultBiasSetting_ = sistrip::invalid_; //@@ 

// ----------------------------------------------------------------------------
// 
const float OptoScanAnalysis::fedAdcGain_ = 1.024 / 1024.; // [V/ADC]

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::optoScanAnalysis_),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    baseSlope_(4,sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis() 
  : CommissioningAnalysis(sistrip::optoScanAnalysis_),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    baseSlope_(4,sistrip::invalid_)
{;}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::reset() {
  gain_       = sistrip::invalid_; 
  bias_       = VInt(4,sistrip::invalid_); 
  measGain_   = VFloat(4,sistrip::invalid_); 
  zeroLight_  = VFloat(4,sistrip::invalid_); 
  linkNoise_  = VFloat(4,sistrip::invalid_);
  liftOff_    = VFloat(4,sistrip::invalid_); 
  threshold_  = VFloat(4,sistrip::invalid_); 
  tickHeight_ = VFloat(4,sistrip::invalid_);
  baseSlope_  = VFloat(4,sistrip::invalid_);
}
  

// ----------------------------------------------------------------------------
// 
bool OptoScanAnalysis::isValid() const {
  return ( gain_ < sistrip::maximum_ &&
	   bias_[gain_] < sistrip::maximum_ &&
	   getErrorCodes().empty() );
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );

  std::stringstream extra1,extra2,extra3; 
  extra1 << sistrip::extrainfo::gain_ << gain() << sistrip::extrainfo::digital_ << "0";
  extra2 << sistrip::extrainfo::gain_ << gain() << sistrip::extrainfo::digital_ << "1";
  extra3 << sistrip::extrainfo::gain_ << gain() << sistrip::extrainfo::baselineRms_;
  
  std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra1.str() ).title();
  std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra2.str() ).title();
  std::string title3 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra3.str() ).title();
  
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
     << title1 << ";" << title2 << ";" << title3
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::print( std::stringstream& ss, uint32_t gain ) { 

  if ( gain >= 4 ) { gain = gain_; }

  if ( gain >= bias_.size() ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected gain setting: " << gain;
    return;
  }

  header( ss );
  ss <<  std::fixed << std::setprecision(2)
     << " Optimum LLD gain setting : " << gain_ << std::endl
     << " LLD gain setting         : " << gain << std::endl
     << " LLD bias setting         : " << bias_[gain] << std::endl
     << " Measured gain      [V/V] : " << measGain_[gain] << std::endl
     << " Zero light level   [ADC] : " << zeroLight_[gain] << std::endl
     << " Link noise [ADC]         : " << linkNoise_[gain] << std::endl
     << " Baseline 'lift off' [mA] : " << liftOff_[gain] << std::endl
     << " Laser threshold     [mA] : " << threshold_[gain] << std::endl
     << " Tick mark height   [ADC] : " << tickHeight_[gain] << std::endl
     << " Baseline slope [ADC/I2C] : " << baseSlope_[gain] << std::endl
     << std::boolalpha 
     << " isValid                  : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ")   : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;

}
