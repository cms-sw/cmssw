/** \class CSCALCTStatusDigi
 * 
 *  Digi for CSC ALCT info available in DDU
 *
 *
 */
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"

#include <ostream>
#include <cstring>
#include <cstdint>

CSCALCTStatusDigi::CSCALCTStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =8;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}

std::ostream & operator<<(std::ostream & o, const CSCALCTStatusDigi& digi) {
  o << " ";  
  o <<"\n";

  return o;
}

