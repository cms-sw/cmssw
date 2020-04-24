#ifndef MagESector_H
#define MagESector_H

/** \class MagESector
 *  A sector of volumes in the endcap.
 *  One sector is composed of several layers (MagELayer)
 *
 *  \author N. Amapane - INFN Torino
 */

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>

class MagVolume;
class MagELayer; 

class MagESector {
public:
  /// Constructor
  MagESector(std::vector<MagELayer*>& layers, Geom::Phi<float> phiMin);

  /// Destructor
  virtual ~MagESector();

  /// Find the volume containing a point, with a given tolerance
  const MagVolume * findVolume(const GlobalPoint & gp, double tolerance) const;

  /// Phi of sector start
  Geom::Phi<float> minPhi() const {return thePhiMin;}

private:
  std::vector<MagELayer*> theLayers;
  Geom::Phi<float> thePhiMin;
};
#endif

