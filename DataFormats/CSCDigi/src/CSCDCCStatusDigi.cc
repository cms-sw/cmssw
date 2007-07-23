/** \class CSCDCCStatusDigi
 * 
 *  Digi for CSC DCC info available in DDU
 *
 *  $Date: 2007/05/21 20:06:55 $
 *  $Revision: 1.1 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h>
#include <bitset>

using namespace std;

CSCDCCStatusDigi::CSCDCCStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
