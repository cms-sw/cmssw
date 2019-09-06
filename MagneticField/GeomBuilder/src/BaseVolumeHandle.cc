/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "BaseVolumeHandle.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"

#include "MagneticField/Layers/interface/MagVerbosity.h"

#include <cassert>
#include <string>
#include <iterator>
#include <iomanip>
#include <iostream>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;

BaseVolumeHandle::BaseVolumeHandle(bool expand2Pi, bool debugVal)
    : magVolume(nullptr),
      masterSector(1),
      theRN(0.),
      theRMin(0.),
      theRMax(0.),
      refPlane(nullptr),
      expand(expand2Pi),
      isIronFlag(false),
      debug(debugVal) {}

BaseVolumeHandle::~BaseVolumeHandle() {
  if (refPlane != nullptr) {
    delete refPlane;
    refPlane = nullptr;
  }
}

const Surface::GlobalPoint &BaseVolumeHandle::center() const { return center_; }

void BaseVolumeHandle::buildPhiZSurf(double startPhi, double deltaPhi, double zhalf, double rCentr) {
  // This is 100% equal for cons and tubs!!!

  GlobalVector planeXAxis = refPlane->toGlobal(LocalVector(1, 0, 0));
  GlobalVector planeYAxis = refPlane->toGlobal(LocalVector(0, 1, 0));
  GlobalVector planeZAxis = refPlane->toGlobal(LocalVector(0, 0, 1));

  // Local Y axis of the faces at +-phi.
  GlobalVector y_phiplus = refPlane->toGlobal(LocalVector(cos(startPhi + deltaPhi), sin(startPhi + deltaPhi), 0.));
  GlobalVector y_phiminus = refPlane->toGlobal(LocalVector(cos(startPhi), sin(startPhi), 0.));

  Surface::RotationType rot_Z(planeXAxis, planeYAxis);
  Surface::RotationType rot_phiplus(planeZAxis, y_phiplus);
  Surface::RotationType rot_phiminus(planeZAxis, y_phiminus);

  GlobalPoint pos_zplus(center_.x(), center_.y(), center_.z() + zhalf);
  GlobalPoint pos_zminus(center_.x(), center_.y(), center_.z() - zhalf);
  // BEWARE: in this case, the origin for phiplus,phiminus surfaces is
  // at radius R and not on a plane passing by center_ orthogonal to the radius.
  GlobalPoint pos_phiplus(
      refPlane->toGlobal(LocalPoint(rCentr * cos(startPhi + deltaPhi), rCentr * sin(startPhi + deltaPhi), 0.)));
  GlobalPoint pos_phiminus(refPlane->toGlobal(LocalPoint(rCentr * cos(startPhi), rCentr * sin(startPhi), 0.)));
  surfaces[zplus] = new Plane(pos_zplus, rot_Z);
  surfaces[zminus] = new Plane(pos_zminus, rot_Z);
  surfaces[phiplus] = new Plane(pos_phiplus, rot_phiplus);
  surfaces[phiminus] = new Plane(pos_phiminus, rot_phiminus);

  if (debug) {
    cout << "Actual Center at: " << center_ << " R " << center_.perp() << " phi " << center_.phi() << endl;
    cout << "RN            " << theRN << endl;

    cout << "pos_zplus    " << pos_zplus << " " << pos_zplus.perp() << " " << pos_zplus.phi() << endl
         << "pos_zminus   " << pos_zminus << " " << pos_zminus.perp() << " " << pos_zminus.phi() << endl
         << "pos_phiplus  " << pos_phiplus << " " << pos_phiplus.perp() << " " << pos_phiplus.phi() << endl
         << "pos_phiminus " << pos_phiminus << " " << pos_phiminus.perp() << " " << pos_phiminus.phi() << endl;

    cout << "y_phiplus  " << y_phiplus << endl;
    cout << "y_phiminus " << y_phiminus << endl;

    cout << "rot_Z    " << surfaces[zplus]->toGlobal(LocalVector(0., 0., 1.)) << endl
         << "rot_phi+ " << surfaces[phiplus]->toGlobal(LocalVector(0., 0., 1.)) << " phi "
         << surfaces[phiplus]->toGlobal(LocalVector(0., 0., 1.)).phi() << endl
         << "rot_phi- " << surfaces[phiminus]->toGlobal(LocalVector(0., 0., 1.)) << " phi "
         << surfaces[phiminus]->toGlobal(LocalVector(0., 0., 1.)).phi() << endl;
  }

  //   // Check ordering.
  if (debug) {
    if (pos_zplus.z() < pos_zminus.z()) {
      cout << "*** WARNING: pos_zplus < pos_zminus " << endl;
    }
    if (Geom::Phi<float>(pos_phiplus.phi() - pos_phiminus.phi()) < 0.) {
      cout << "*** WARNING: pos_phiplus < pos_phiminus " << endl;
    }
  }
}

bool BaseVolumeHandle::sameSurface(const Surface &s1, Sides which_side, float tolerance) {
  //Check for null comparison
  if (&s1 == (surfaces[which_side]).get()) {
    if (debug)
      cout << "      sameSurface: OK (same ptr)" << endl;
    return true;
  }

  const float maxtilt = 0.999;

  const Surface &s2 = *(surfaces[which_side]);
  // Try with a plane.
  const Plane *p1 = dynamic_cast<const Plane *>(&s1);
  if (p1 != nullptr) {
    const Plane *p2 = dynamic_cast<const Plane *>(&s2);
    if (p2 == nullptr) {
      if (debug)
        cout << "      sameSurface: different types" << endl;
      return false;
    }

    if ((fabs(p1->normalVector().dot(p2->normalVector())) > maxtilt) &&
        (fabs((p1->toLocal(p2->position())).z()) < tolerance)) {
      if (debug)
        cout << "      sameSurface: OK " << fabs(p1->normalVector().dot(p2->normalVector())) << " "
             << fabs((p1->toLocal(p2->position())).z()) << endl;
      return true;
    } else {
      if (debug)
        cout << "      sameSurface: not the same: " << p1->normalVector() << p1->position() << endl
             << "                                 " << p2->normalVector() << p2->position() << endl
             << fabs(p1->normalVector().dot(p2->normalVector())) << " " << (p1->toLocal(p2->position())).z() << endl;
      return false;
    }
  }

  // Try with a cylinder.
  const Cylinder *cy1 = dynamic_cast<const Cylinder *>(&s1);
  if (cy1 != nullptr) {
    const Cylinder *cy2 = dynamic_cast<const Cylinder *>(&s2);
    if (cy2 == nullptr) {
      if (debug)
        cout << "      sameSurface: different types" << endl;
      return false;
    }
    // Assume axis is the same!
    if (fabs(cy1->radius() - cy2->radius()) < tolerance) {
      return true;
    } else {
      return false;
    }
  }

  // Try with a cone.
  const Cone *co1 = dynamic_cast<const Cone *>(&s1);
  if (co1 != nullptr) {
    const Cone *co2 = dynamic_cast<const Cone *>(&s2);
    if (co2 == nullptr) {
      if (debug)
        cout << "      sameSurface: different types" << endl;
      return false;
    }
    // FIXME
    if (fabs(co1->openingAngle() - co2->openingAngle()) < maxtilt &&
        (co1->vertex() - co2->vertex()).mag() < tolerance) {
      return true;
    } else {
      return false;
    }
  }

  if (debug)
    cout << "      sameSurface: unknown surfaces..." << endl;
  return false;
}

bool BaseVolumeHandle::setSurface(const Surface &s1, Sides which_side) {
  //Check for null assignment
  if (&s1 == (surfaces[which_side]).get()) {
    isAssigned[which_side] = true;
    return true;
  }

  if (!sameSurface(s1, which_side)) {
    cout << "***ERROR: setSurface: trying to assign a surface that does not match destination surface. Skipping."
         << endl;
    const Surface &s2 = *(surfaces[which_side]);
    //FIXME: Just planes for the time being!!!
    const Plane *p1 = dynamic_cast<const Plane *>(&s1);
    const Plane *p2 = dynamic_cast<const Plane *>(&s2);
    if (p1 != nullptr && p2 != nullptr)
      cout << p1->normalVector() << p1->position() << endl << p2->normalVector() << p2->position() << endl;
    return false;
  }

  if (isAssigned[which_side]) {
    if (&s1 != (surfaces[which_side]).get()) {
      cout << "*** WARNING BaseVolumeHandle::setSurface: trying to reassign a surface to a different surface instance"
           << endl;
      return false;
    }
  } else {
    surfaces[which_side] = &s1;
    isAssigned[which_side] = true;
    if (debug)
      cout << "     Volume " << name << " # " << copyno << " Assigned: " << (int)which_side << endl;
    return true;
  }

  return false;  // let the compiler be happy
}

const Surface &BaseVolumeHandle::surface(Sides which_side) const { return *(surfaces[which_side]); }

const Surface &BaseVolumeHandle::surface(int which_side) const {
  assert(which_side >= 0 && which_side < 6);
  return *(surfaces[which_side]);
}
