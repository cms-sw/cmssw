// Last commit: $Id: SiStripNullKey.cc,v 1.2 2007/03/21 08:22:59 bainbrid Exp $

#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripNullKey::SiStripNullKey() : SiStripKey() {;}

// -----------------------------------------------------------------------------
// 
bool SiStripNullKey::isEqual( const SiStripKey& input ) const {
  SiStripKey& temp = const_cast<SiStripKey&>(input);
  if ( &dynamic_cast<SiStripNullKey&>(temp) ) { return true; }
  else { return false; }
}

// -----------------------------------------------------------------------------
// 
bool SiStripNullKey::isConsistent( const SiStripKey& input ) const {
  return isEqual(input);
}

// -----------------------------------------------------------------------------
//
bool SiStripNullKey::isValid() const { 
  return false;
}

// -----------------------------------------------------------------------------
//
bool SiStripNullKey::isValid( const sistrip::Granularity& gran ) const {
  return false; 
}

// -----------------------------------------------------------------------------
//
bool SiStripNullKey::isInvalid() const { 
  return true;
}

// -----------------------------------------------------------------------------
//
bool SiStripNullKey::isInvalid( const sistrip::Granularity& gran ) const {
  return true;
}

// -----------------------------------------------------------------------------
// 
void SiStripNullKey::initFromValue() {;}

// -----------------------------------------------------------------------------
//
void SiStripNullKey::initFromKey() {;}

// -----------------------------------------------------------------------------
// 
void SiStripNullKey::initFromPath() {;}

// -----------------------------------------------------------------------------
// 
void SiStripNullKey::initGranularity() {;}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripNullKey& input ) {
  return os << std::endl
	    << " [SiStripNullKey::print]" << std::endl
	    << std::hex
	    << " 32-bit key  : 0x" 
	    << std::setfill('0') 
	    << std::setw(8) << input.key() << std::endl
	    << std::setfill(' ') 
	    << std::dec
	    << " Directory   : " << input.path() << std::endl
	    << " Granularity : "
	    << SiStripEnumsAndStrings::granularity( input.granularity() ) << std::endl
 	    << " Channel     : " << input.channel() << std::endl
	    << " isValid    : " << input.isValid();
}
