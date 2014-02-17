
/** \Class RPCMaskReClusterizer
 *  $Date: 2008/10/14 08:52:31 $
 *  $Revision: 1.1 $
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "RPCMaskReClusterizer.h"



RPCMaskReClusterizer::RPCMaskReClusterizer()
{

}


RPCMaskReClusterizer::~RPCMaskReClusterizer()
{

}


RPCClusterContainer RPCMaskReClusterizer::doAction(const RPCDetId& id,
                                                    RPCClusterContainer& initClusters,
                                                    const RollMask& mask)
{

  RPCClusterContainer finClusters;
  RPCCluster prev;

  unsigned int j = 0;


  for (RPCClusterContainer::const_iterator i = initClusters.begin(); i != initClusters.end(); i++ ) {

    RPCCluster cl = *i;

    if ( i == initClusters.begin() ) {
      prev = cl;
      j++;
      if ( j == initClusters.size() ) {
	finClusters.insert(prev);
      }
      else if ( j < initClusters.size() ) {
	continue;
      }
    }


    if ( ((prev.firstStrip()-cl.lastStrip()) == 2 && this->get(mask,(cl.lastStrip()+1)))
	 && (cl.bx() == prev.bx()) ) {

      RPCCluster merged(cl.firstStrip(),prev.lastStrip(),cl.bx());
      prev = merged;
      j++;
      if ( j == initClusters.size() ) {
	finClusters.insert(prev);
      }
    }

    else {

      j++;
      if ( j < initClusters.size() ) {
	finClusters.insert(prev);
	prev = cl;
      }
      if ( j == initClusters.size() ) {
	finClusters.insert(prev);
	finClusters.insert(cl);
      }
    }
 
  }

  return finClusters;

}



int RPCMaskReClusterizer::get(const RollMask& mask, int strip)
{

  if ( mask.test(strip-1) ) return 1;
  else return 0;

}
