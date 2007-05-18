/** \class CSCDMBStatusDigi
 * 
 *  Digi for CSC DMB info available in DDU
 *
 *  $Date: 2007/04/04 14:40:58 $
 *  $Revision: 1.3 $
 *
 */
#include <DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h>
#include <iostream>
#include <bitset>
#include <boost/cstdint.hpp>

using namespace std;

CSCDMBStatusDigi::CSCDMBStatusDigi(uint16_t * header, uint16_t * trailer)
{
  uint sizeInBytes =16;
  memcpy(header_, header, sizeInBytes);
  memcpy(trailer_, trailer, sizeInBytes);
}
