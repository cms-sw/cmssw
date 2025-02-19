/** \class CSCTMBStatusDigi
 * 
 *  Digi for CSC TMB info available in DDU
 *
 *  $Date: 2009/05/09 20:23:34 $
 *  $Revision: 1.9 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include <ostream>
#include <cstring>
#include <stdint.h>

CSCTMBStatusDigi::CSCTMBStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =54;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}

std::ostream & operator<<(std::ostream & o, const CSCTMBStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

