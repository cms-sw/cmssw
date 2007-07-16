#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundSpan.h"


void BoundSurface::computePhiSpan() {
  m_phiSpan =  boundSpan::computePhiSpan(*this);
}
