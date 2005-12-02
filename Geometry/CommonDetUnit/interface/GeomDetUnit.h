#ifndef CommonDet_GeomDetUnit_H
#define CommonDet_GeomDetUnit_H

#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class Topology;
//class Readout;
class GeomDetType;

class GeomDetUnit : public GeomDet {
public:

  explicit GeomDetUnit( BoundPlane* sp);

  virtual ~GeomDetUnit();
  
  virtual const BoundSurface& surface() const;
  virtual const BoundPlane&   specificSurface() const { return *thePlane;}

  virtual const Topology& topology() const = 0;

  virtual const GeomDetType& type() const = 0;


  virtual DetId geographicalId() const = 0;

  /// DetUnit does not have components
  virtual std::vector< const GeomDet*> components() const {
    return std::vector< const GeomDet*>();
  }

private:

  ReferenceCountingPointer<BoundPlane>  thePlane;

  // alignment part of interface available only to friend 
  friend class AlignableDetUnit;
  void move( const Surface::PositionType& displacement);
  void rotate( const Surface::RotationType& rotation);
  void setPosition( const Surface::PositionType& position, 
                    const Surface::RotationType& rotation);

};
  
#endif




