/** \class CSCTMBStatusDigi
 * 
 *  Digi for CSC TMB info available in DDU
 *
 *  $Date: 2008/02/12 17:39:52 $
 *  $Revision: 1.6 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include <iostream>

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

