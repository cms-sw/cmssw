#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/GeomExceptions.h"

float Bounds::significanceInside(const Local3DPoint&, const LocalError&) const {
  throw GeometryError("howMuchInside not implemented");
}
