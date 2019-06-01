#ifndef BeamSpotGeomDet_H
#define BeamSpotGeomDet_H

/** \class BeamSpotGeomDet
 *
 * A GeomDet used to create transient tracking rec hits for the
 * beam spot. The DetId originates from a static memebr function
 * in AlignableBeamSpot.
 *
 * Author     : Andreas Mussgiller
 * date       : 2010/08/30
 * last update: $Date: 2010/09/10 12:02:44 $
 * by         : $Author: mussgill $
 */

#include <iostream>

#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

class BeamSpotGeomDet : public GeomDet {
public:
  typedef GeomDetEnumerators::SubDetector SubDetector;

  explicit BeamSpotGeomDet(const ReferenceCountingPointer<BoundPlane>& plane) : GeomDet(plane) {
    setDetId(AlignableBeamSpot::detId());
  }

  ~BeamSpotGeomDet() override {}

  SubDetector subDetector() const override { return GeomDetEnumerators::invalidDet; }

  std::vector<const GeomDet*> components() const override { return std::vector<const GeomDet*>(); }
};

#endif
