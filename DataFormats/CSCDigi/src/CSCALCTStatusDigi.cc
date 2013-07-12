/** \class CSCALCTStatusDigi
 * 
 *  Digi for CSC ALCT info available in DDU
 *
 *  $Date: 2009/01/30 23:22:47 $
 *  $Revision: 1.5 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"

#include <ostream>
#include <cstring>
#include <stdint.h>

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

