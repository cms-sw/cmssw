/** \class CSCDDUStatusDigi
 * 
 *  Digi for CSC DDU info available in DDU
 *
 *  $Date: 2008/10/29 18:34:41 $
 *  $Revision: 1.4 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include <ostream>
#include <cstring>

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

