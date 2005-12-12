#ifndef CommonDet_GeomDet_H
#define CommonDet_GeomDet_H

#include "Geometry/Surface/interface/BoundPlane.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

class Topology;
class GeomDetType;
class AlignmentPositionError;

/** Base class for GeomDetUNit and for composite GeomDets, used by RecHit.
 */

class GeomDet {
public:

  virtual ~GeomDet() {}
  
  virtual const BoundSurface& surface() const = 0;
  virtual const BoundPlane&   specificSurface() const = 0;

  virtual DetId geographicalId() const = 0;

  /** Return pointer to alignment errors (defaults to "null" if not 
   *  reimplemented in the derived classes) */
  virtual AlignmentPositionError* alignmentPositionError() const { return 0;}

  /// Returns direct components, if any
  virtual std::vector< const GeomDet*> components() const = 0;

private:

  ReferenceCountingPointer<BoundPlane>  thePlane;

  // alignment part of interface available only to friend 
  friend class AlignableDetUnit;

  /// Relative displacement (with respect to current position)
  void move( const Surface::PositionType& displacement);

  /// Relative rotation (with respect to current orientation)
  void rotate( const Surface::RotationType& rotation);

  /** Replaces the current position and rotation with new ones; actually replaces the 
   *  surface with a new surface.
   */
  void setPosition( const Surface::PositionType& position, 
                    const Surface::RotationType& rotation);

  /** create the AlignmentPositionError for this Det if not existing yet,
   *  or replace the existing one by the given one. For adding, use the
   *  +=,-=  methods of the AlignmentPositionError*/
  virtual void setAlignmentPositionError (const AlignmentPositionError& ape) {} 

};
  
#endif




