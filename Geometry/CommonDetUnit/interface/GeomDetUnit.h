#ifndef CommonDet_GeomDetUnit_H
#define CommonDet_GeomDetUnit_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class Topology;
//class Readout;
class GeomDetType;

class GeomDetUnit : public GeomDet {
public:

  explicit GeomDetUnit( BoundPlane* sp);
  explicit GeomDetUnit( const ReferenceCountingPointer<BoundPlane>& plane);

  virtual ~GeomDetUnit();
  
  virtual const Topology& topology() const = 0;

  virtual const GeomDetType& type() const = 0;

  /// DetUnit does not have components
  virtual std::vector< const GeomDet*> components() const {
    return std::vector< const GeomDet*>();
  }

  virtual const GeomDet* component(DetId id) const {return 0;}

  // Which subdetector
  virtual SubDetector subDetector() const;

};
  
#endif




