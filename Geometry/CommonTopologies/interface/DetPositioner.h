#ifndef DetPositioner_H
#define DetPositioner_H

#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"

/* A base class for classes which modify the positions/orientations of GeomDets.
 * The derived classes can call the methods moveGeomDet, rotateGeomDet, setGeomDetPosition,
 * setAlignmentPositionError and setSurfaceDeformation to change the position, orientation etc.
 */

class DetPositioner {
public:
  virtual ~DetPositioner() {}

protected:
  void moveGeomDet(GeomDet& det, const GlobalVector& displacement) { det.move(displacement); }

  /** Relative rotation (with respect to current orientation)
   * Does not move components (if any).
   */
  void rotateGeomDet(GeomDet& det, const Surface::RotationType& rotation) { det.rotate(rotation); }

  /** Replaces the current position and rotation with new ones; actually replaces the 
   *  surface with a new surface.
   *  Does not move components (if any).
   */
  void setGeomDetPosition(GeomDet& det, const Surface::PositionType& position, const Surface::RotationType& rotation) {
    det.setPosition(position, rotation);
  }

  /** create the AlignmentPositionError for this Det if not existing yet,
   *  or replace the existing one by the given one. For adding, use the
   *  +=,-=  methods of the AlignmentPositionError
   *  Does not affect the AlignmentPositionError of components (if any).
   */
  bool setAlignmentPositionError(GeomDet& det, const AlignmentPositionError& ape) {
    return det.setAlignmentPositionError(ape);
  }

  /** set the SurfaceDeformation for this DetUnit.
   *  Does not affect the SurfaceDeformation of components (if any).
   */
  void setSurfaceDeformation(GeomDetUnit& detUnit, const SurfaceDeformation* deformation) {
    detUnit.setSurfaceDeformation(deformation);
  }
};

#endif
