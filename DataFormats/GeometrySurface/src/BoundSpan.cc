#include "DataFormats/GeometrySurface/interface/BoundSpan.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

void BoundSpan::compute(Surface const& plane) {
  const TrapezoidalPlaneBounds* trapezoidalBounds(dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds(dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  Surface::GlobalPoint corners[8];

  if (trapezoidalBounds) {
    std::array<const float, 4> const& parameters = (*trapezoidalBounds).parameters();

    auto hbotedge = parameters[0];
    auto htopedge = parameters[1];
    auto hapothem = parameters[3];
    auto thickness = (*trapezoidalBounds).thickness();

    corners[0] = plane.toGlobal(LocalPoint(-htopedge, hapothem, thickness / 2));
    corners[1] = plane.toGlobal(LocalPoint(htopedge, hapothem, thickness / 2));
    corners[2] = plane.toGlobal(LocalPoint(hbotedge, -hapothem, thickness / 2));
    corners[3] = plane.toGlobal(LocalPoint(-hbotedge, -hapothem, thickness / 2));
    corners[4] = plane.toGlobal(LocalPoint(-htopedge, hapothem, -thickness / 2));
    corners[5] = plane.toGlobal(LocalPoint(htopedge, hapothem, -thickness / 2));
    corners[6] = plane.toGlobal(LocalPoint(hbotedge, -hapothem, -thickness / 2));
    corners[7] = plane.toGlobal(LocalPoint(-hbotedge, -hapothem, -thickness / 2));

  } else if (rectangularBounds) {
    auto length = rectangularBounds->length();
    auto width = rectangularBounds->width();
    auto thickness = (*rectangularBounds).thickness();

    corners[0] = plane.toGlobal(LocalPoint(-width / 2, -length / 2, thickness / 2));
    corners[1] = plane.toGlobal(LocalPoint(-width / 2, +length / 2, thickness / 2));
    corners[2] = plane.toGlobal(LocalPoint(+width / 2, -length / 2, thickness / 2));
    corners[3] = plane.toGlobal(LocalPoint(+width / 2, +length / 2, thickness / 2));
    corners[4] = plane.toGlobal(LocalPoint(-width / 2, -length / 2, -thickness / 2));
    corners[5] = plane.toGlobal(LocalPoint(-width / 2, +length / 2, -thickness / 2));
    corners[6] = plane.toGlobal(LocalPoint(+width / 2, -length / 2, -thickness / 2));
    corners[7] = plane.toGlobal(LocalPoint(+width / 2, +length / 2, -thickness / 2));
  } else {
  }

  float phimin = corners[0].barePhi();
  float phimax = phimin;
  float zmin = corners[0].z();
  float zmax = zmin;
  float rmin = corners[0].perp2();
  float rmax = rmin;
  for (int i = 1; i < 8; i++) {
    auto cPhi = corners[i].barePhi();
    if (Geom::phiLess(cPhi, phimin)) {
      phimin = cPhi;
    }
    if (Geom::phiLess(phimax, cPhi)) {
      phimax = cPhi;
    }
    auto z = corners[i].z();
    if (z < zmin)
      zmin = z;
    if (z > zmax)
      zmax = z;
    auto r = corners[i].perp2();
    if (r < rmin)
      rmin = r;
    if (r > rmax)
      rmax = r;
  }
  m_zSpan.first = zmin;
  m_zSpan.second = zmax;
  m_rSpan.first = std::sqrt(rmin);
  m_rSpan.second = std::sqrt(rmax);
  m_phiSpan.first = phimin;
  m_phiSpan.second = phimax;
}
