
/** \Class GEMMaskReClusterizer
 *  $Date: 2013/04/24 17:16:34 $
 *  $Revision: 1.1 $
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "GEMMaskReClusterizer.h"



GEMMaskReClusterizer::GEMMaskReClusterizer()
{

}


GEMMaskReClusterizer::~GEMMaskReClusterizer()
{

}


GEMClusterContainer GEMMaskReClusterizer::doAction(const GEMDetId& id,
                                                    GEMClusterContainer& initClusters,
                                                    const EtaPartitionMask& mask)
{

  GEMClusterContainer finClusters;
  GEMCluster prev;

  unsigned int j = 0;


  for (GEMClusterContainer::const_iterator i = initClusters.begin(); i != initClusters.end(); i++ ) {

    GEMCluster cl = *i;

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

      GEMCluster merged(cl.firstStrip(),prev.lastStrip(),cl.bx());
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



int GEMMaskReClusterizer::get(const EtaPartitionMask& mask, int strip)
{

  if ( mask.test(strip-1) ) return 1;
  else return 0;

}
