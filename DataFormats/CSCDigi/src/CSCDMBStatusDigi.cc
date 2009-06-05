/** \class CSCDMBStatusDigi
 * 
 *  Digi for CSC DMB info available in DDU
 *
 *  $Date: 2007/07/23 12:08:20 $
 *  $Revision: 1.3 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h>

using namespace std;

CSCDMBStatusDigi::CSCDMBStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =16;
  uint16_t trailerSizeInBytes =16;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
