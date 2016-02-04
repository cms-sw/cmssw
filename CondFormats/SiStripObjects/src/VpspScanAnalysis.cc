#include "CondFormats/SiStripObjects/interface/VpspScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
VpspScanAnalysis::VpspScanAnalysis( const uint32_t& key )
  : CommissioningAnalysis(key,"VpspScanAnalysis"),
    vpsp_(2,sistrip::invalid_), 
    adcLevel_(2,sistrip::invalid_),
    fraction_(2,sistrip::invalid_),
    topEdge_(2,sistrip::invalid_),
    bottomEdge_(2,sistrip::invalid_),
    topLevel_(2,sistrip::invalid_),
    bottomLevel_(2,sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
//
VpspScanAnalysis::VpspScanAnalysis()
  : CommissioningAnalysis("VpspScanAnalysis"),
    vpsp_(2,sistrip::invalid_), 
    adcLevel_(2,sistrip::invalid_),
    fraction_(2,sistrip::invalid_),
    topEdge_(2,sistrip::invalid_),
    bottomEdge_(2,sistrip::invalid_),
    topLevel_(2,sistrip::invalid_),
    bottomLevel_(2,sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
//
void VpspScanAnalysis::reset() {
  vpsp_ = VInt(2,sistrip::invalid_);
}

// ----------------------------------------------------------------------------
// 
bool VpspScanAnalysis::isValid() const {
  return ( vpsp_[0] < 1. * sistrip::valid_ &&
	   vpsp_[1] < 1. * sistrip::valid_ &&
	   adcLevel_[0] < 1. * sistrip::valid_ &&
	   adcLevel_[1] < 1. * sistrip::valid_ &&
	   topLevel_[0] < 1. * sistrip::valid_ &&
	   topLevel_[1] < 1. * sistrip::valid_ &&
	   bottomLevel_[0] < 1. * sistrip::valid_ &&
	   bottomLevel_[1] < 1. * sistrip::valid_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  
  std::stringstream extra1,extra2;
  extra1 << sistrip::apv_ << "0";
  extra2 << sistrip::apv_ << "1";
  
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
     << title1 << ";" << title2
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void VpspScanAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  header( ss );
  ss << " Monitorables for APV : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss <<  std::fixed << std::setprecision(2)
     << " VPSP setting         : " << vpsp_[iapv] << std::endl 
     << " Signal level   [ADC] : " << adcLevel_[iapv] << std::endl
     << " Fraction         [%] : " << "(N/A)" /*fraction_[iapv]*/ << std::endl
     << " Top edge       [bin] : " << topEdge_[iapv] << std::endl
     << " Bottom edge    [bin] : " << bottomEdge_[iapv] << std::endl
     << " Top level      [ADC] : " << topLevel_[iapv] << std::endl
     << " Bottom level   [ADC] : " << bottomLevel_[iapv] << std::endl
     << std::boolalpha 
     << " isValid              : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << "): ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}






