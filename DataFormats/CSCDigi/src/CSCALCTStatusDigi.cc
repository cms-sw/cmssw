/** \class CSCALCTStatusDigi
 * 
 *  Digi for CSC ALCT info available in DDU
 *
 *  $Date: 2007/05/21 20:06:55 $
 *  $Revision: 1.1 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h>
#include <bitset>

using namespace std;

CSCALCTStatusDigi::CSCALCTStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =8;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
