/** \class CSCDDUStatusDigi
 * 
 *  Digi for CSC DDU info available in DDU
 *
 *  $Date: 2008/02/12 17:40:17 $
 *  $Revision: 1.3 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"

#include <iostream>

CSCDDUStatusDigi::CSCDDUStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =24;
  uint16_t trailerSizeInBytes =24;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}

std::ostream & operator<<(std::ostream & o, const CSCDDUStatusDigi& digi) {
  o << " "; 
  o <<"\n";
 
  return o;
}

