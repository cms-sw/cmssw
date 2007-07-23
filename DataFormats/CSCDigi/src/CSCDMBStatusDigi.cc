/** \class CSCDMBStatusDigi
 * 
 *  Digi for CSC DMB info available in DDU
 *
 *  $Date: 2007/05/21 20:05:07 $
 *  $Revision: 1.2 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h>
#include <bitset>

using namespace std;

CSCDMBStatusDigi::CSCDMBStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
