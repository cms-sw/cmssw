#ifndef MagBSlab_H
#define MagBSlab_H

/** \class MagBSlab
 *  
 *  A  container of volumes in the barrel. It is part of the hierarchical 
 *  organisation of barrel volumes:
 *
 *  A barrel layer (MagBLayer) groups volumes at the same distance to
 *  the origin. It consists of 12 sectors in phi (MagBSector). 
 *  Each sector consists of one or more rods (MagBRods) of equal width in phi.
 *  Rods consist of one or more slabs (MagBSlab); each one consisting of one 
 *  or, in few cases, several volumes with the same lenght in Z.
 *
 *  $Date: 2013/05/30 21:57:39 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */

#include <vector>

class MagVolume;
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MagBSlab {
public:
  /// Constructor
  MagBSlab(const std::vector<MagVolume*>& volumes, double zMin);

  /// Destructor
  virtual ~MagBSlab();

  /// Find the volume containing a point, with a given tolerance
  MagVolume * findVolume(const GlobalPoint & gp, double tolerance) const;

  /// Lower Z bound
  double minZ() const { return theZMin;}
  
private:
  std::vector<MagVolume*> theVolumes;
  double theZMin;
  
};
#endif

