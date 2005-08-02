/**
   \file
   Impl of DTDetId

   \author Stefano ARGIRO
   \version $Id$
   \date 02 Aug 2005
*/

static const char CVSId[] = "$Id$";

#include <iostream>
#include <DataFormats/DetId/interface/DTDetId.h>

std::ostream& operator<<( std::ostream& os, const DTDetId& id ){

  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     << " La:"<< id.layer()<<" ";

  return os;
}


