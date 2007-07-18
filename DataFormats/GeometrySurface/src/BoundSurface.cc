#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundSpan.h"


void BoundSurface::computeSpan() {
  boundSpan::computeSpan(*this);
}
