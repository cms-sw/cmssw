// Last commit: $Id: SiStripKey.cc,v 1.2 2007/03/21 08:22:59 bainbrid Exp $

#include "DataFormats/SiStripCommon/interface/SiStripKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iomanip>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripKey::SiStripKey( const uint32_t& key ) :
  key_(key),
  path_(sistrip::null_),
  granularity_(sistrip::UNDEFINED_GRAN),
  channel_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
// 
SiStripKey::SiStripKey( const std::string& path ) :
  key_(sistrip::invalid32_),
  path_(path),
  granularity_(sistrip::UNDEFINED_GRAN),
  channel_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
// 
SiStripKey::SiStripKey() : 
  key_(sistrip::invalid32_),
  path_(sistrip::null_),
  granularity_(sistrip::UNDEFINED_GRAN),
  channel_(sistrip::invalid_)
{;}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripKey& input ) {
  return os << std::endl
	    << " [SiStripKey::print]" << std::endl
	    << std::hex
	    << " 32-bit key  : 0x" 
	    << std::setfill('0') 
	    << std::setw(8) << input.key() << std::endl
	    << std::setfill(' ') 
	    << std::dec
	    << " Directory   : " << input.path() << std::endl
	    << " Granularity : " 
	    << SiStripEnumsAndStrings::granularity( input.granularity() ) << std::endl
	    << " Channel     : " << input.channel();
  
}
