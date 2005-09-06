#ifndef MagELayer_H
#define MagELayer_H

/** \class MagELayer
 *  A layer of volumes in an endcap sector.
 *
 *  $Date: 2004/06/22 17:05:13 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - INFN Torino
 */

#include "Geometry/Vector/interface/GlobalPoint.h"

#include <vector>

class MagVolume;

class MagELayer {
public:
  /// Constructor
  MagELayer(std::vector<MagVolume*> volumes, double zMin, double zMax);

  /// Destructor
  virtual ~MagELayer();

  /// Find the volume containing a point, with a given tolerance
  MagVolume * findVolume(const GlobalPoint & gp, double tolerance) const;

  /// Lower Z bound
  double minZ() const {return theZMin;}

  /// Upper Z bound
  double maxZ() const {return theZMax;}

private:
  std::vector<MagVolume*> theVolumes;
  double theZMin;
  double theZMax;
};
#endif

