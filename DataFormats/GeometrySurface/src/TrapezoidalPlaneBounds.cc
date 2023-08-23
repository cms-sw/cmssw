

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include <cmath>

TrapezoidalPlaneBounds::TrapezoidalPlaneBounds(float be, float te, float a, float t)
    : hbotedge(be), htopedge(te), hapothem(a), hthickness(t) {
  // pre-compute offset of triangle vertex and tg of (half) opening
  // angle of the trapezoid for faster inside() implementation.

  offset = a * (te + be) / (te - be);  // check sign if te < be !!!
  tan_a = te / std::abs(offset + a);
}

int TrapezoidalPlaneBounds::yAxisOrientation() const { return (hbotedge > htopedge) ? -1 : 1; }

bool TrapezoidalPlaneBounds::inside(const Local2DPoint& p) const {
  return (std::abs(p.y()) < hapothem) && (std::abs(p.x()) < tan_a * std::abs(p.y() + offset));
}

bool TrapezoidalPlaneBounds::inside(const Local3DPoint& p) const {
  return ((std::abs(p.y()) < hapothem) && (std::abs(p.z()) < hthickness)) &&
         std::abs(p.x()) < tan_a * std::abs(p.y() + offset);
}

bool TrapezoidalPlaneBounds::inside(const Local3DPoint& p, const LocalError& err, float scale) const {
  if (scale >= 0 && inside(p))
    return true;

  TrapezoidalPlaneBounds tmp(hbotedge + std::sqrt(err.xx()) * scale,
                             htopedge + std::sqrt(err.xx()) * scale,
                             hapothem + std::sqrt(err.yy()) * scale,
                             hthickness);
  return tmp.inside(p);
}

bool TrapezoidalPlaneBounds::inside(const Local2DPoint& p, const LocalError& err, float scale) const {
  return Bounds::inside(p, err, scale);
}

float TrapezoidalPlaneBounds::significanceInside(const Local3DPoint& p, const LocalError& err) const {
  return std::max((std::abs(p.y()) - hapothem) / std::sqrt(err.yy()),
                  (std::abs(p.x()) - tan_a * std::abs(p.y() + offset)) / std::sqrt(err.xx()));
}

Bounds* TrapezoidalPlaneBounds::clone() const { return new TrapezoidalPlaneBounds(*this); }

const std::array<const float, 4> TrapezoidalPlaneBounds::parameters() const {
  // Same order as geant3 for constructor compatibility
  std::array<const float, 4> vec{{hbotedge, htopedge, hthickness, hapothem}};
  return vec;
}
