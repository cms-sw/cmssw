/** \class CSCDCCStatusDigi
 * 
 *  Digi for CSC DCC info available in DDU
 *
 *  $Date: 2008/02/12 17:40:25 $
 *  $Revision: 1.3 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h>

using namespace std;

CSCDCCStatusDigi::CSCDCCStatusDigi(const uint16_t * header, const uint16_t * trailer, const uint32_t & error) 
{
  errorFlag_=error;
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
