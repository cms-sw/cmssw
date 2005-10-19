/**
   \file
   Impl of RPCDetId

   \author Stefano ARGIRO
   \version $Id: RPCDetId.cc,v 1.2 2005/08/23 09:11:28 argiro Exp $
   \date 02 Aug 2005
*/

static const char CVSId[] = "$Id: RPCDetId.cc,v 1.2 2005/08/23 09:11:28 argiro Exp $";

#include <iostream>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

RPCDetId::RPCDetId():DetId(0){}

std::ostream& operator<<( std::ostream& os, const RPCDetId& id ){

// do differently whether it's station or Wheel.

  os << " Ro:"<< id.roll()
     << " Co:"<< id.copy() 
     << " Se:"<< id.sector()
     << " Pl:"<< id.plane()
     << " Et:"<< id.eta()<<" ";

  return os;
}


