// Hep-LIKE geometrical 3D LINE class
//
//
// Author: BKH
//

#ifndef HepLine3D_hh
#define HepLine3D_hh

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Normal3D.h"
#include "CLHEP/Geometry/Plane3D.h"
#include <iostream>

class HepLine3D {
protected:
  HepGeom::Point3D<double> pp;
  HepGeom::Vector3D<double> uu;
  double eps;

public:
  HepLine3D(const HepGeom::Point3D<double>& p, const HepGeom::Vector3D<double>& v, double sml = 1.e-10)
      : pp(p), uu(v * (v.mag() > 1.e-10 ? 1. / v.mag() : 1)), eps(fabs(sml)) {}

  HepLine3D(const HepGeom::Point3D<double>& p1, const HepGeom::Point3D<double>& p2, double sml = 1.e-10)
      : pp(p1), uu((p2 - p1) * ((p2 - p1).mag() > 1.e-10 ? 1. / (p2 - p1).mag() : 1)), eps(fabs(sml)) {}

  // Copy constructor
  HepLine3D(const HepLine3D& line) : pp(line.pp), uu(line.uu), eps(line.eps) {}

  // Destructor
  ~HepLine3D(){};

  // Assignment
  HepLine3D& operator=(const HepLine3D& line) {
    pp = line.pp;
    uu = line.uu;
    eps = line.eps;
    return *this;
  }

  // Test for equality
  bool operator==(const HepLine3D& l) const { return pp == l.pp && uu == l.uu; }

  // Test for inequality
  bool operator!=(const HepLine3D& l) const { return pp != l.pp || uu != l.uu; }

  const HepGeom::Point3D<double>& pt() const { return pp; }

  const HepGeom::Vector3D<double>& uv() const { return uu; }

  HepGeom::Point3D<double> point(const HepGeom::Plane3D<double>& pl, bool& parallel) const {
    const double num(-pl.d() - pl.a() * pp.x() - pl.b() * pp.y() - pl.c() * pp.z());
    const double den(pl.a() * uu.x() + pl.b() * uu.y() + pl.c() * uu.z());

    parallel = (eps > fabs(num)) || (eps > fabs(den));

    return (parallel ? pp : HepGeom::Point3D<double>(pp + uu * (num / den)));
  }

  HepGeom::Point3D<double> point(const HepGeom::Point3D<double>& q) const {
    return (pp + ((q.x() - pp.x()) * uu.x() + (q.y() - pp.y()) * uu.y() + (q.z() - pp.z()) * uu.z()) * uu);
  }

  double dist2(const HepGeom::Point3D<double>& q) const { return (q - point(q)).mag2(); }
  double dist(const HepGeom::Point3D<double>& q) const { return (q - point(q)).mag(); }
};

#endif
