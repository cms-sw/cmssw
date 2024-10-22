//   COCOA class implementation file
//Id:  LightRay.cc
//CAT: Fit
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include <cstdlib>
#include <cmath>  // include floating-point std::abs functions

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ construct a default LightRay
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
LightRay::LightRay() {
  _point = CLHEP::Hep3Vector(0., 0., 0.);
  _direction = CLHEP::Hep3Vector(0., 0., 1.);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Set the point and direction to that of the laser or source
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::startLightRay(OpticalObject* opto) {
  if (ALIUtils::debug >= 3)
    std::cout << std::endl << "LR: CREATE LIGHTRAY " << opto->name() << " OptO type is " << opto->type() << std::endl;

  //---------- Get Z axis of opto
  CLHEP::Hep3Vector ZAxis(0., 0., 1.);
  const CLHEP::HepRotation& rmt = opto->rmGlob();
  ZAxis = rmt * ZAxis;

  //---------- By convention, direction of LightRay = opto_ZAxis
  setDirection(ZAxis);
  setPoint(opto->centreGlob());

  if (ALIUtils::debug >= 3) {
    dumpData(" LightRay at creation ");
  }
  if (ALIUtils::debug >= 5) {
    ALIUtils::dumprm(rmt, "laser Rotation matrix");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
LightRay::LightRay(OpticalObject* opto1, OpticalObject* opto2) {
  if (ALIUtils::debug >= 7)
    std::cout << std::endl << "LR:CREATE LIGHTRAY FROM SOURCE" << opto2->name() << std::endl;

  CLHEP::Hep3Vector _ZAxis(0., 0., 1.);
  //-  LightRay* linetmp;
  //-linetmp = new LightRay;
  //---------- set direction and point
  setDirection(opto2->centreGlob() - opto1->centreGlob());
  setPoint(opto1->centreGlob());

  if (ALIUtils::debug >= 9)
    std::cout << "OPT" << opto1 << opto1->name() << std::endl;
  //-  std::cout << "centre glob" << &p1->aff()->centre_glob() << std::endl;
  if (ALIUtils::debug >= 9) {
    dumpData(" ");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
LightRay::LightRay(CLHEP::Hep3Vector& vec1, CLHEP::Hep3Vector& vec2) {
  CLHEP::Hep3Vector dir = vec2 - vec1;
  dir *= 1. / dir.mag();
  setDirection(dir);
  setPoint(vec1);
  if (ALIUtils::debug >= 9) {
    dumpData(" ");
  }
}

//@@ intersect: Intersect a LightRay with a plane and change thePoint to the intersection point
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::intersect(const OpticalObject& opto) {
  if (ALIUtils::debug >= 3)
    std::cout << "% LR INTERSECT WITH OPTO" << std::endl;
  CLHEP::Hep3Vector ZAxis(0., 0., 1.);
  const CLHEP::HepRotation& rmt = opto.rmGlob();
  ZAxis = rmt * ZAxis;
  ALIPlane optoPlane(opto.centreGlob(), ZAxis);
  intersect(optoPlane);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ intersect: Intersect a LightRay with a plane and change thePoint to the intersection point
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::intersect(const ALIPlane& plane) {
  if (ALIUtils::debug >= 4)
    std::cout << "% LR INTERSECT WITH PLANE" << std::endl;
  if (ALIUtils::debug >= 4) {
    ALIUtils::dump3v(plane.point(), "plane point");
    ALIUtils::dump3v(plane.normal(), "plane normal");
    //-    dumpData(" ");
  }

  //---------- Check that they intersect
  if (std::abs(plane.normal() * direction()) < 1.E-10) {
    std::cerr << " !!!! INTERSECTION NOT POSSIBLE: LightRay is perpendicular to plane " << std::endl;
    std::cerr << " plane.normal()*direction() = " << plane.normal() * direction() << std::endl;
    ALIUtils::dump3v(direction(), "LightRay direction ");
    ALIUtils::dump3v(plane.normal(), "plane normal ");
    exit(1);
  }

  //---------- Get intersection point between LightRay and plane
  CLHEP::Hep3Vector vtemp = plane.point() - _point;
  if (ALIUtils::debug >= 5)
    ALIUtils::dump3v(vtemp, "n_r = point  - point_plane");
  ALIdouble dtemp = _direction * plane.normal();
  if (ALIUtils::debug >= 5)
    std::cout << " lightray* plate normal" << dtemp << std::endl;
  if (dtemp != 0.) {
    dtemp = (vtemp * plane.normal()) / dtemp;
    if (ALIUtils::debug >= 5)
      std::cout << " n_r*plate normal" << dtemp << std::endl;
  } else {
    std::cerr << "!!! LightRay: Intersect With Plane: plane and light ray parallel: no intersection" << std::endl;
  }
  vtemp = _direction * dtemp;
  if (ALIUtils::debug >= 5)
    ALIUtils::dump3v(vtemp, "n_r scaled");
  CLHEP::Hep3Vector inters = vtemp + _point;
  if (ALIUtils::debug >= 3)
    ALIUtils::dump3v(inters, "INTERSECTION point ");

  _point = inters;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Intersect the LightRay with a plane and then change the direction from reflection on this plane
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::reflect(const ALIPlane& plane) {
  intersect(plane);
  if (ALIUtils::debug >= 4)
    std::cout << "% LR: REFLECT IN PLANE " << std::endl;
  ALIdouble cosang = -(plane.normal() * _direction) / plane.normal().mag() / _direction.mag();
  if (ALIUtils::debug >= 5) {
    std::cout << " cosang = " << cosang << std::endl;
    ALIUtils::dump3v(plane.normal(), " plane normal");
  }
  _direction += plane.normal() * 2 * cosang;
  if (ALIUtils::debug >= 5)
    dumpData("LightRay after reflection: ");
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@Deviate a LightRay because of refraction when it passes from a medium of refraction index  refra_ind1 to a medium of refraction index  refra_ind2
//@@ 3D deviation: it actually deviates in the plane of the plate normal and lightray, in the other plane there is no deviation
//@@ Refract inside this plane
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::refract(const ALIPlane& plate, const ALIdouble refra_ind1, const ALIdouble refra_ind2) {
  if (ALIUtils::debug >= 5) {
    std::cout << "% LR REFRACT: "
              << "refra_ind1 = " << refra_ind1 << " refra_ind2 = " << refra_ind2 << std::endl;
    std::cout << "@ First intersect with plate plane " << std::endl;
  }

  intersect(plate);

  //---------- First plane: formed by plate normal and lightray, but get two ortonormal std::vectors in this plane (one of it plate normal)
  CLHEP::Hep3Vector Axis1 = plate.normal().cross(direction());
  //----- Check lightray is not parallel to plate normal
  if (Axis1.mag() < 1.E-6) {
    if (ALIUtils::debug >= 3) {
      std::cout << " light ray normal to plane, no refraction " << std::endl;
    }
    if (ALIUtils::debug >= 2) {
      dumpData("LightRay after refraction: ");
    }

    return;
  }

  if (ALIUtils::debug >= 5) {
    ALIUtils::dump3v(Axis1, " axis 1 temp ");
  }
  Axis1 = Axis1.cross(plate.normal());
  Axis1 *= 1. / Axis1.mag();
  //----- Project lightray on this plane
  if (ALIUtils::debug >= 4) {
    ALIUtils::dump3v(plate.normal(), " plate normal ");
    ALIUtils::dump3v(Axis1, " axis 1 ");
  }

  //----- Angle between LightRay and plate_normal before traversing
  ALIdouble cosang = -(plate.normal() * direction()) / plate.normal().mag() / direction().mag();
  ALIdouble sinang = sqrt(1. - cosang * cosang);

  //----- Angle between LightRay projection and plate normal after traversing (refracted)
  ALIdouble sinangp = sinang * refra_ind1 / refra_ind2;
  if (std::abs(sinangp) > 1.) {
    std::cerr << " !!!EXITING LightRay::refract: incidence ray on plane too close to face, refraction will not allow "
                 "entering "
              << std::endl;
    ALIUtils::dump3v(plate.normal(), " plate normal ");
    ALIUtils::dump3v(direction(), " light ray direction ");
    std::cout << " refraction index first medium " << refra_ind1 << " refraction index second medium " << refra_ind2
              << std::endl;
    exit(1);
  }

  if (ALIUtils::debug >= 4) {
    std::cout << "LightRay refract on plane 1: sin(ang) before = " << sinang << " sinang after " << sinangp
              << std::endl;
  }
  ALIdouble cosangp = sqrt(1. - sinangp * sinangp);
  //----- Change Lightray direction in this plane
  //--- Get sign of projections in plate normal and axis1
  ALIdouble signN = direction() * plate.normal();
  signN /= std::abs(signN);
  ALIdouble sign1 = direction() * Axis1;
  sign1 /= std::abs(sign1);
  if (ALIUtils::debug >= 4) {
    dumpData("LightRay refract: direction before plate");
    std::cout << " sign projection on plate normal " << signN << " sign projection on Axis1 " << sign1 << std::endl;
  }
  setDirection(signN * cosangp * plate.normal() + sign1 * sinangp * Axis1);
  //-  std::cout << " " << signN  << " " << cosangp  << " " << plate.normal() << " " << sign1  << " " << sinangp   << " " << Axis1 << std::endl;

  if (ALIUtils::debug >= 3) {
    dumpData("LightRay refract: direction after plate");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ shift and deviates around X, Y and Z of opto
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::shiftAndDeviateWhileTraversing(const OpticalObject* opto, char behav) {
  ALIstring ename("devi X");
  ename[4] = behav;
  ename[5] = 'X';
  ALIdouble deviX = opto->findExtraEntryValue(ename);
  ename[5] = 'Y';
  ALIdouble deviY = opto->findExtraEntryValue(ename);
  ename[5] = 'Z';
  ALIdouble deviZ = opto->findExtraEntryValue(ename);

  ename = "shift X";
  ename[5] = behav;
  ename[6] = 'X';
  ALIdouble shiftX = opto->findExtraEntryValue(ename);
  ename[6] = 'Y';
  ALIdouble shiftY = opto->findExtraEntryValue(ename);
  ename[6] = 'Z';
  ALIdouble shiftZ = opto->findExtraEntryValue(ename);

  if (ALIUtils::debug >= 3) {
    //-    std::cout << " shift X " << shiftX << " shiftY " << shiftY << " shiftZ " << shiftZ << std::endl;
    //-    std::cout << " deviX " << deviX << " deviY " << deviY << " deviZ " << deviZ << std::endl;
    std::cout << " shift X " << shiftX << " shift Y " << shiftY << std::endl;
    std::cout << " devi X " << deviX << " devi Y " << deviY << std::endl;
  }

  shiftAndDeviateWhileTraversing(opto, shiftX, shiftY, shiftZ, deviX, deviY, deviZ);
  //  shiftAndDeviateWhileTraversing( shiftX, shiftY, deviX, deviY );
}

void LightRay::shiftAndDeviateWhileTraversing(const OpticalObject* opto,
                                              ALIdouble shiftX,
                                              ALIdouble shiftY,
                                              ALIdouble shiftZ,
                                              ALIdouble deviX,
                                              ALIdouble deviY,
                                              ALIdouble deviZ) {
  //----- Get local opto X, Y and Z axis
  CLHEP::Hep3Vector XAxis(1., 0., 0.);
  CLHEP::Hep3Vector YAxis(0., 1., 0.);
  CLHEP::Hep3Vector ZAxis(0., 0., 1.);
  const CLHEP::HepRotation& rmt = opto->rmGlob();
  XAxis = rmt * XAxis;
  YAxis = rmt * YAxis;
  ZAxis = rmt * ZAxis;

  if (ALIUtils::debug >= 5) {
    ALIUtils::dump3v(XAxis, "X axis of opto");
    ALIUtils::dump3v(YAxis, "Y axis of opto");
    ALIUtils::dump3v(ZAxis, "Z axis of opto");
  }

  //---------- Shift
  CLHEP::Hep3Vector pointold = _point;
  _point += shiftX * XAxis;
  _point += shiftY * YAxis;
  _point += shiftZ * ZAxis;
  if (_point != pointold && ALIUtils::debug >= 3) {
    ALIUtils::dump3v(_point - pointold, "CHANGE point");
  }

  //---------- Deviate
  CLHEP::Hep3Vector direcold = _direction;
  if (ALIUtils::debug >= 5) {
    ALIUtils::dump3v(XAxis, "XAxis");
    ALIUtils::dump3v(YAxis, "YAxis");
    ALIUtils::dump3v(ZAxis, "ZAxis");
    ALIUtils::dump3v(_direction, "LightRay direction");
  }

  _direction.rotate(deviX, XAxis);
  if (_direction != direcold && ALIUtils::debug >= 3) {
    std::cout << " deviX " << deviX << std::endl;
    ALIUtils::dump3v(_direction - direcold, "CHANGE direction");
  }
  _direction.rotate(deviY, YAxis);
  if (_direction != direcold && ALIUtils::debug >= 3) {
    std::cout << " deviY " << deviY << std::endl;
    ALIUtils::dump3v(_direction - direcold, "CHANGE direction");
  }
  _direction.rotate(deviZ, ZAxis);
  if (_direction != direcold && ALIUtils::debug >= 3) {
    std::cout << " deviZ " << deviZ << std::endl;
    ALIUtils::dump3v(_direction - direcold, "CHANGE direction");
  }

  if (_direction != direcold && ALIUtils::debug >= 3) {
    ALIUtils::dump3v(_direction - direcold, "CHANGE direction");
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ shift and deviates around two directions perpendicular to LightRay
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
/*void LightRay::shiftAndDeviateWhileTraversing( ALIdouble shiftAxis1, ALIdouble shiftAxis2, ALIdouble deviAxis1, ALIdouble deviAxis2 )
{
  if(ALIUtils::debug >= 3) {
    std::cout << " shift Axis 1 " << shiftAxis1 << " shift Axis 2 " << shiftAxis2 << std::endl;
    std::cout << " devi Axis 1 " << deviAxis1 << " devi Axis 2 " << deviAxis2 << std::endl;
  }

  //----- Get two directions perpendicular to LightRay
  //-- First can be (y,-x,0), unless direciton is (0,0,1), or close
  CLHEP::Hep3Vector PerpAxis1;
  if(std::abs(std::abs(_direction.z())-1.) > 0.1) {
    if (ALIUtils::debug >= 99) ALIUtils::dump3v( _direction, "_direction1");
    PerpAxis1 = CLHEP::Hep3Vector(_direction.y(),-_direction.x(),0.);
  } else {
    if (ALIUtils::debug >= 99) ALIUtils::dump3v( _direction, "_direction2");
    PerpAxis1 = CLHEP::Hep3Vector(_direction.z(),0.,-_direction.y());
  }
  if (ALIUtils::debug >= 4) ALIUtils::dump3v( PerpAxis1, "1st perpendicular direction of DDet");

  CLHEP::Hep3Vector PerpAxis2 = _direction.cross(PerpAxis1);
  if (ALIUtils::debug >= 4) ALIUtils::dump3v( PerpAxis2, "2nd perpendicular direction of DDet");

  //---------- Shift
  CLHEP::Hep3Vector pointold = _point;
  _point += shiftAxis1*PerpAxis1;
  _point += shiftAxis2*PerpAxis2;
  if(_point != pointold && ALIUtils::debug >= 3 ) {
    ALIUtils::dump3v( _point-pointold, "CHANGE point");
  }

  //---------- Deviate
  CLHEP::Hep3Vector direcold = _direction;
  _direction.rotate(deviAxis1, PerpAxis1);
  _direction.rotate(deviAxis2, PerpAxis2);
  if(_direction != direcold && ALIUtils::debug >= 3) {
    ALIUtils::dump3v( _direction-direcold, "CHANGE direction");
  }

}
*/

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void LightRay::dumpData(const ALIstring& str) const {
  std::cout << str << std::endl;
  ALIUtils::dump3v(_point, "$$ LightRay point: ");
  ALIUtils::dump3v(_direction, "$$ LightRay direction: ");
  /*
  CLHEP::Hep3Vector dirn = _direction;
  dirn.rotateZ( -23.72876*3.1415926/180.);
  ALIUtils::dump3v( dirn, "$$ LightRay direction: ");
  */
}
