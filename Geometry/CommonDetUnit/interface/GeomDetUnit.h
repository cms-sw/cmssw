#ifndef CommonDet_GeomDetUnit_H
#define CommonDet_GeomDetUnit_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class Topology;
//class Readout;
class GeomDetType;
class SurfaceDeformation;

class GeomDetUnit : public GeomDet {
public:

  explicit GeomDetUnit( Plane* sp);
  explicit GeomDetUnit( const ReferenceCountingPointer<Plane>& plane);

  virtual ~GeomDetUnit();
  
  virtual const Topology& topology() const = 0;

  virtual const GeomDetType& type() const = 0;

  /// DetUnit does not have components
  virtual std::vector< const GeomDet*> components() const {
    return std::vector< const GeomDet*>();
  }

  virtual const GeomDet* component(DetId /*id*/) const {return 0;}

  // Which subdetector
  virtual SubDetector subDetector() const;

  /// Return pointer to surface deformation. 
  /// Defaults to "null" if not reimplemented in the derived classes.
  virtual const SurfaceDeformation* surfaceDeformation() const { return 0; }
  
private:

  /// Alignment part of interface, available only to friend 
  friend class DetPositioner;

  /// set the SurfaceDeformation for this GeomDetUnit.
  /// Does not affect the SurfaceDeformation of components (if any).
  /// Throws if not implemented in derived class.
  virtual void setSurfaceDeformation(const SurfaceDeformation * deformation);
};

#endif




