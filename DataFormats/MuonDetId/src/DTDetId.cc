/** \file
 * Impl of DTDetId
 *
 * \author Stefano ARGIRO
 * \version $Id: DTDetId.cc,v 1.2 2005/08/23 09:11:28 argiro Exp $
 * \date 02 Aug 2005
*/

static const char CVSId[] = "$Id: DTDetId.cc,v 1.2 2005/08/23 09:11:28 argiro Exp $";

#include <iostream>
#include <DataFormats/MuonDetId/interface/DTDetId.h>

DTDetId::DTDetId():DetId(0){}

std::ostream& operator<<( std::ostream& os, const DTDetId& id ){

  os << " Wh:"<< id.wheel()
     << " St:"<< id.station() 
     << " Se:"<< id.sector()
     << " Sl:"<< id.superlayer()
     << " La:"<< id.layer()
     << " Wi:"<< id.wire()
     <<" ";

  return os;
}


