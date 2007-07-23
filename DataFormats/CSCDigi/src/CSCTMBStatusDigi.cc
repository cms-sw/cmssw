/** \class CSCTMBStatusDigi
 * 
 *  Digi for CSC TMB info available in DDU
 *
 *  $Date: 2007/05/21 20:05:07 $
 *  $Revision: 1.4 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h>
#include <bitset>

using namespace std;

CSCTMBStatusDigi::CSCTMBStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =54;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
