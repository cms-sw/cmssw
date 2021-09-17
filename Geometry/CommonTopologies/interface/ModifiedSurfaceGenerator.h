#ifndef ModifiedSurfaceGenerator_h_
#define ModifiedSurfaceGenerator_h_

#include "DataFormats/GeometrySurface/interface/Surface.h"

/** Creates a new instance of a BoundSurface at 
 *  a new location (using the copy constructor). */

template <class T>
class ConstReferenceCountingPointer;
template <class T>
class ReferenceCountingPointer;
class MediumProperties;

template <class T>
class ModifiedSurfaceGenerator {
private:
  typedef ReferenceCountingPointer<T> SurfacePointer;

public:
  /// constructor from pointer
  ModifiedSurfaceGenerator(const T* surface) : theSurface(surface) {}
  /// constructor from ReferenceCountingPointer
  ModifiedSurfaceGenerator(const SurfacePointer surface) : theSurface(surface.get()) {}
  /** creation of a new surface at a different position, but with
   *  identical Bounds and MediumProperties */
  SurfacePointer atNewPosition(const Surface::PositionType& position, const Surface::RotationType& rotation) const {
    const MediumProperties& mp = theSurface->mediumProperties();
    SurfacePointer newSurface(new T(position, rotation, mp, theSurface->bounds().clone()));
    return newSurface;
  }

private:
  /// original surface
  ConstReferenceCountingPointer<T> theSurface;
};

#endif
