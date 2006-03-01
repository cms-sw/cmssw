#ifndef CommonDet_GeomDet_H
#define CommonDet_GeomDet_H

/** \class GeomDet
 *  Base class for GeomDetUnit and for composite GeomDet s. 
 *
 *  $Date: $
 *  $Revision: $
 */


#include "Geometry/Surface/interface/BoundPlane.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

class AlignmentPositionError;

class GeomDet {
public:

  // empty constructor, requires a call to setSurface from derived class constructor
  // GeomDet();

  explicit GeomDet( BoundPlane* plane);

  explicit GeomDet( const ReferenceCountingPointer<BoundPlane>& plane);

  virtual ~GeomDet();
  
  virtual const BoundPlane& surface() const {return *thePlane;}
  virtual const BoundPlane& specificSurface() const {return *thePlane;} // obsolete?

  virtual DetId geographicalId() const = 0;

  /// Return pointer to alignment errors. 
  /// Defaults to "null" if not reimplemented in the derived classes.
  virtual AlignmentPositionError* alignmentPositionError() const { return 0;}

  /// Returns direct components, if any
  virtual std::vector< const GeomDet*> components() const = 0;

protected:

  // setSurface( const ReferenceCountingPointer<BoundPlane>& plane);

private:

  ReferenceCountingPointer<BoundPlane>  thePlane;
  AlignmentPositionError*               theAlignmentPositionError;

  /// Alignment part of interface, available only to friend 
  friend class AlignableDetUnit;

  /// Relative displacement (with respect to current position).
  /// Does not move components (if any).
  void move( const GlobalVector& displacement);

  /// Relative rotation (with respect to current orientation).
  /// Does not move components (if any).
  void rotate( const Surface::RotationType& rotation);

  /// Replaces the current position and rotation with new ones.
  /// actually replaces the surface with a new surface.
  /// Does not move components (if any).
   
  void setPosition( const Surface::PositionType& position, 
		    const Surface::RotationType& rotation);

  /// create the AlignmentPositionError for this Det if not existing yet,
  /// or replace the existing one by the given one. For adding, use the
  /// +=,-=  methods of the AlignmentPositionError
  /// Does not affect the AlignmentPositionError of components (if any).
  
  virtual void setAlignmentPositionError (const AlignmentPositionError& ape); 

};
  
#endif




