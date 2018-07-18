#include "DataFormats/DetId/interface/DetId.h"

DetId gluedId( const DetId& du) 
{
  unsigned int mask = ~3; // mask the last two bits
  return DetId( du.rawId() & mask);
}
