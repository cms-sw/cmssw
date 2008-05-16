#include "CondFormats/SiStripObjects/interface/PedsOnlyAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
PedsOnlyAnalysis::PedsOnlyAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"PedsOnlyAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)), 
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    legacy_(false)
{}

// ----------------------------------------------------------------------------
// 
PedsOnlyAnalysis::PedsOnlyAnalysis() 
  : CommissioningAnalysis("PedsOnlyAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)), 
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    legacy_(false)
{}

// ----------------------------------------------------------------------------
// 
void PedsOnlyAnalysis::reset() {
  peds_        = VVFloat(2,VFloat(128,sistrip::invalid_)); 
  raw_         = VVFloat(2,VFloat(128,sistrip::invalid_));
  pedsMean_    = VFloat(2,sistrip::invalid_); 
  pedsSpread_  = VFloat(2,sistrip::invalid_); 
  rawMean_     = VFloat(2,sistrip::invalid_);
  rawSpread_   = VFloat(2,sistrip::invalid_);
  pedsMax_     = VFloat(2,sistrip::invalid_); 
  pedsMin_     = VFloat(2,sistrip::invalid_); 
  rawMax_      = VFloat(2,sistrip::invalid_);
  rawMin_      = VFloat(2,sistrip::invalid_);
  legacy_      = false;
}

// ----------------------------------------------------------------------------
// 
bool PedsOnlyAnalysis::isValid() const {
  return ( pedsMean_[0] < sistrip::maximum_ &&
	   pedsMean_[1] < sistrip::maximum_ &&
	   pedsSpread_[0] < sistrip::maximum_ &&
	   pedsSpread_[1] < sistrip::maximum_ &&
	   rawMean_[0] < sistrip::maximum_ &&
	   rawMean_[1] < sistrip::maximum_ &&
	   rawSpread_[0] < sistrip::maximum_ &&
	   rawSpread_[1] < sistrip::maximum_ &&
	   pedsMax_[0] < sistrip::maximum_ &&
	   pedsMax_[1] < sistrip::maximum_ &&
	   pedsMin_[0] < sistrip::maximum_ &&
	   pedsMin_[1] < sistrip::maximum_ &&
	   rawMax_[0] < sistrip::maximum_ &&
	   rawMax_[1] < sistrip::maximum_ &&
	   rawMin_[0] < sistrip::maximum_ &&
	   rawMin_[1] < sistrip::maximum_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void PedsOnlyAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  
  std::stringstream extra1,extra2,extra3;
  if ( legacy_ ) {
    extra1 << sistrip::extrainfo::pedsAndRawNoise_; 
    extra2 << sistrip::extrainfo::pedsAndCmSubNoise_; 
    extra3 << sistrip::extrainfo::commonMode_;
  } else {
    extra1 << sistrip::extrainfo::pedestals_; 
    extra2 << sistrip::extrainfo::rawNoise_; 
    extra3 << sistrip::extrainfo::commonMode_;
  }
  
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
					  sistrip::APV, 
					  SiStripFecKey::i2cAddr( fec_key.lldChan(), true ),
					  extra3.str() ).title();
  std::string title4 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::APV, 
					  SiStripFecKey::i2cAddr( fec_key.lldChan(), false ),
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
     << title1 << ";" << title2 << ";" << title3 << ";" << title4
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void PedsOnlyAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 

  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  
  if ( peds_[iapv].size() < 128 ||
       raw_[iapv].size() < 128 ) { 
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of pedestal/noise values: " 
      << peds_[iapv].size() << ", " 
      << raw_[iapv].size();
    return;
  }
  
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss << std::fixed << std::setprecision(2);
  ss << " Example peds/noise for strips   : "
     << "     0,     31,     63,    127" << std::endl
     << "  Peds                     [ADC] : " 
     << std::setw(6) << peds_[iapv][0] << ", " 
     << std::setw(6) << peds_[iapv][31] << ", " 
     << std::setw(6) << peds_[iapv][63] << ", " 
     << std::setw(6) << peds_[iapv][127] << std::endl
     << "  Raw noise                [ADC] : " 
     << std::setw(6) << raw_[iapv][0] << ", " 
     << std::setw(6) << raw_[iapv][31] << ", " 
     << std::setw(6) << raw_[iapv][63] << ", " 
     << std::setw(6) << raw_[iapv][127] << std::endl;
  ss << " Mean peds +/- spread      [ADC] : " << pedsMean_[iapv] << " +/- " << pedsSpread_[iapv] << std::endl 
     << " Min/Max pedestal          [ADC] : " << pedsMin_[iapv] << " <-> " << pedsMax_[iapv] << std::endl
     << " Mean raw noise +/- spread [ADC] : " << rawMean_[iapv] << " +/- " << rawSpread_[iapv] << std::endl 
     << " Min/Max raw noise         [ADC] : " << rawMin_[iapv] << " <-> " << rawMax_[iapv] << std::endl
     << std::boolalpha 
     << " isValid                         : " << isValid()  << std::endl
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

