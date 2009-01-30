/** \class CSCTMBStatusDigi
 * 
 *  Digi for CSC TMB info available in DDU
 *
 *  $Date: 2008/10/29 18:34:41 $
 *  $Revision: 1.7 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include <ostream>
#include <cstring>

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

