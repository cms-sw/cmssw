/** \class CSCDDUStatusDigi
 * 
 *  Digi for CSC DDU info available in DDU
 *
 *  $Date: 2007/05/21 20:06:55 $
 *  $Revision: 1.1 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h>
#include <bitset>

using namespace std;

CSCDDUStatusDigi::CSCDDUStatusDigi(const uint16_t * header, const uint16_t * trailer)
{
  uint16_t headerSizeInBytes =24;
  uint16_t trailerSizeInBytes =24;
  memcpy(header_, header, headerSizeInBytes);
  memcpy(trailer_, trailer, trailerSizeInBytes);
}
