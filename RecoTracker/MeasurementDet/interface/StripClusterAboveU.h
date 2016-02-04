#ifndef StripClusterAboveU_H
#define StripClusterAboveU_H

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

/** Function object that selects StripClusters who's first coordinate 
 *  in the measurement frame is bigger than some value.
 *  Looks obsolete.
 */

class StripClusterAboveU {
public:
  StripClusterAboveU( float u) : theU(u) {}
  bool operator()( const SiStripCluster& hit) const {
    return hit.barycenter() > theU; 
  }
private:
  float theU;
};

#endif 
