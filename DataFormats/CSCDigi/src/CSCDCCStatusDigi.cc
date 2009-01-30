/** \class CSCDCCStatusDigi
 * 
 *  Digi for CSC DCC info available in DDU
 *
 *  $Date: 2008/10/29 18:34:41 $
 *  $Revision: 1.5 $
 *
 */
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"
#include <ostream>
#include <cstring>

CSCDCCStatusDigi::CSCDCCStatusDigi(const uint16_t * header, const uint16_t * trailer, const uint32_t & error) 
{
  errorFlag_=error;
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}

std::ostream & operator<<(std::ostream & o, const CSCDCCStatusDigi& digi) {
  o << " ";
  o <<"\n";

  return o;
}

