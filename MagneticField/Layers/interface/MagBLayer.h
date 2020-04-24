#ifndef MagBLayer_H
#define MagBLayer_H

/** \class MagBLayer
 *  
 *  A barrel layer (MagBLayer) groups volumes at the same distance to
 *  the origin. It consists of either 1 single volume (a cylinder) or
 *  12 sectors in phi (MagBSector). 
 *  Each sector consists of one or more rods (MagBRods) of equal width in phi.
 *  Rods consist of one or more slabs (MagBSlab); each one consisting of one 
 *  or, in few cases, several volumes with the same lenght in Z.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class MagBSector;
class MagVolume;
template <class T> class PeriodicBinFinderInPhi;

class MagBLayer {
public:
  /// Constructor
  MagBLayer(std::vector<MagBSector*>& sectors, double rMin);

  /// Constructor for a trivial layer consisting of one single volume.
  MagBLayer(MagVolume* aVolume, double rMin);

  /// Destructor
  virtual ~MagBLayer();

  /// Find the volume containing a point, with a given tolerance
  const MagVolume * findVolume(const GlobalPoint & gp, double tolerance) const;
  
  /// Lowest radius of the layer
  double minR() const {return theRMin;}

private:
  // To support either the case of a simple one-volume layer or a
  // composite structure we have both theSectors or theSingleVolume.
  // Only one can be active at a time; not very elegant, but acceptable.
  std::vector<MagBSector*> theSectors;
  MagVolume* theSingleVolume; 
  double theRMin;

  PeriodicBinFinderInPhi<float> * theBinFinder;

};
#endif

