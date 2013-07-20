// Last commit: $Id: SiStripNullKey.cc,v 1.5 2012/07/04 19:04:53 eulisse Exp $

#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>

// -----------------------------------------------------------------------------
// 
SiStripNullKey::SiStripNullKey() : SiStripKey() {;}

// -----------------------------------------------------------------------------
// 
bool SiStripNullKey::isEqual( const SiStripKey& input ) const {
  if ( &dynamic_cast<const SiStripNullKey&>(input) ) { return true; }
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
void SiStripNullKey::print( std::stringstream& ss ) const {
  ss << " [SiStripNullKey::print]" << std::endl
     << std::hex
     << " 32-bit key  : 0x" 
     << std::setfill('0') 
     << std::setw(8) << key() << std::endl
     << std::setfill(' ') 
     << std::dec
     << " Directory   : " << path() << std::endl
     << " Granularity : "
     << SiStripEnumsAndStrings::granularity( granularity() ) << std::endl
     << " Channel     : " << channel() << std::endl
     << " isValid    : " << isValid();
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripNullKey& input ) {
  std::stringstream ss;
  input.print(ss);
  os << ss.str();
  return os;
}
