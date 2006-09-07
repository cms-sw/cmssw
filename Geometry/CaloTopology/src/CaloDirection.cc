#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include <iostream>

std::ostream & operator<<(std::ostream& o,const CaloDirection& d)
{
  switch(d)
    {
    case NONE:
      o << "NONE";
      break;
    case SOUTH:
      o << "SOUTH";
      break;
    case SOUTHEAST:
      o << "SOUTHEAST";
      break;
    case SOUTHWEST:
      o << "SOUTHWEST";
      break;
    case EAST:
      o << "EAST";
      break;
    case WEST:
      o << "WEST";
      break;
    case NORTHEAST:
      o << "NORTHEAST";
      break;
    case NORTHWEST:
      o << "NORTHWEST";
      break;
    case NORTH:
      o << "NORTH";
      break;
    case DOWN:
      o << "DOWN";
      break;
    case DOWNSOUTH:
      o << "DOWNSOUTH";
      break;
    case DOWNSOUTHEAST:
      o << "DOWNSOUTHEAST";
      break;
    case DOWNSOUTHWEST:
      o << "DOWNSOUTHWEST";
      break;
    case DOWNEAST:
      o << "DOWNEAST";
      break;
    case DOWNWEST:
      o << "DOWNWEST";
      break;
    case DOWNNORTHEAST:
      o << "NORTHEAST";
      break;
    case DOWNNORTHWEST:
      o << "NORTHWEST";
      break;
    case DOWNNORTH:
      o << "DOWNNORTH";
      break;
    case UP:
      o << "UP";
      break;
    case UPSOUTH:
      o << "UPSOUTH";
      break;
    case UPSOUTHEAST:
      o << "UPSOUTHEAST";
      break;
    case UPSOUTHWEST:
      o << "UPSOUTHWEST";
      break;
    case UPEAST:
      o << "UPEAST";
      break;
    case UPWEST:
      o << "UPWEST";
      break;
    case UPNORTHEAST:
      o << "NORTHEAST";
      break;
    case UPNORTHWEST:
      o << "NORTHWEST";
      break;
    case UPNORTH:
      o << "UPNORTH";
      break;
    default:
      //o << static_cast<int>(d);
      break;
    }
 
  return o;
}
