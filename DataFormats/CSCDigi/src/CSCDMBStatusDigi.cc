/** \class CSCDMBStatusDigi
 * 
 *  Digi for CSC DMB info available in DDU
 *
 *  $Date: 2009/01/30 23:22:47 $
 *  $Revision: 1.6 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include <ostream>
#include <cstring>
#include <stdint.h>

CSCDMBStatusDigi::CSCDMBStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}

std::ostream & operator<<(std::ostream & o, const CSCDMBStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

