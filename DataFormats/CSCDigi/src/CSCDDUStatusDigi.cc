/** \class CSCDDUStatusDigi
 * 
 *  Digi for CSC DDU info available in DDU
 *
 *  $Date: 2007/07/23 12:08:20 $
 *  $Revision: 1.2 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h>

using namespace std;

CSCDDUStatusDigi::CSCDDUStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =24;
  uint16_t trailerSizeInBytes =24;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
