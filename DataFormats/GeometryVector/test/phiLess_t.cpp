#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include <cassert>

int main() {
  assert(Geom::phiLess(0.f, 2.f));
  assert(Geom::phiLess(6.f, 0.f));
  assert(Geom::phiLess(3.2f, 0.f));
  assert(Geom::phiLess(3.0f, 3.2f));

  assert(Geom::phiLess(-0.3f, 0.f));
  assert(Geom::phiLess(-0.3f, 0.1f));
  assert(Geom::phiLess(-3.0f, 0.f));
  assert(Geom::phiLess(3.0f, -3.0f));
  assert(Geom::phiLess(0.f, -3.4f));
}
