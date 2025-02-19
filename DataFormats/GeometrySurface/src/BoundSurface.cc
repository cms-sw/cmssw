#include "DataFormats/GeometrySurface/interface/BoundSurface.h"


void BoundSurface::computeSpan() {
  boundSpan::computeSpan(*this);
}
