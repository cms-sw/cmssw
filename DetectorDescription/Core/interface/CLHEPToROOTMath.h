#ifndef DD_CLHEPToROOTMath_h
#define DD_CLHEPToROOTMath_h
#include <iostream>
#include <fstream>
#include <iomanip>

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

/// C++ functors for output and conversion of CLHEP and ROOT::Math

class HepRotOutput {
 public:
  HepRotOutput() { }
  ~HepRotOutput() { }
  void operator() ( const CLHEP::HepRotation& r ) {
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << r.xx();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.xy();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.xz() << " ]" << std::endl;
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << r.yx();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.yy();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.yz() << " ]" << std::endl;
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << r.zx();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.zy();
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.zz() << " ]" << std::endl;
  }
};


/* void hepSetAxes( CLHEP::Hep3Vector& x, CLHEP::Hep3Vector& y, CLHEP::Hep3Vector& z */
/* 		 , double thetaX, double phiX */
/* 		 , double thetaY, double phiY */
/* 		 , double thetaZ, double phiZ ) { */
/*   x[0] = sin(thetaX) * cos(phiX); */
/*   x[1] = sin(thetaX) * sin(phiX); */
/*   x[2] = cos(thetaX); */
/*   y[0] = sin(thetaY) * cos(phiY); */
/*   y[1] = sin(thetaY) * sin(phiY); */
/*   y[2] = cos(thetaY); */
/*   z[0] = sin(thetaZ) * cos(phiZ); */
/*   z[1] = sin(thetaZ) * sin(phiZ); */
/*   z[2] = cos(thetaZ); */
/* } */

/* void hepOutVecs( const CLHEP::Hep3Vector& x, const CLHEP::Hep3Vector& y, const CLHEP::Hep3Vector& z ) { */
/*   std::cout << "Vectors used in construction:" << std::endl; */
/*   std::cout << "x vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[0] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[0] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[0] << std::endl; */
/*   std::cout << "y vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[1] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[1] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[1] << std::endl; */
/*   std::cout << "z vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[2] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[2] */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[2] << std::endl; */
/* } */

class DDRotOutput {
 public:
  DDRotOutput() { }
  ~DDRotOutput() { }
  void operator() ( const DDRotationMatrix& r ) {
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;
    r.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << xx;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << xy;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << xz << " ]" << std::endl;
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << yx;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << yy;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << yz << " ]" << std::endl;
    std::cout << "[ " << std::setw(12) << std::fixed << std::setprecision(5) << zx;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << zy;
    std::cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << zz << " ]"<< std::endl;
  }
};

/* void ddSetAxes ( DD3Vector& x, DD3Vector& y,  DD3Vector& z */
/* 		 , double thetaX, double phiX */
/* 		 , double thetaY, double phiY */
/* 		 , double thetaZ, double phiZ ) { */
/*   x.SetX(sin(thetaX) * cos(phiX)); */
/*   x.SetY(sin(thetaX) * sin(phiX)); */
/*   x.SetZ(cos(thetaX)); */
/*   y.SetX(sin(thetaY) * cos(phiY)); */
/*   y.SetY(sin(thetaY) * sin(phiY)); */
/*   y.SetZ(cos(thetaY)); */
/*   z.SetX(sin(thetaZ) * cos(phiZ)); */
/*   z.SetY(sin(thetaZ) * sin(phiZ)); */
/*   z.SetZ(cos(thetaZ)); */
/* } */

/* void ddOutVecs( const DD3Vector& x, const DD3Vector& y, const DD3Vector& z ) { */
/*   std::cout << "Vectors used in construction:" << std::endl; */
/*   std::cout << "x vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x.X() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Y() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Z() << std::endl; */
/*   std::cout << "y vector = " << std::setw(12) << std::fixed << std::setprecision(5) << y.X() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Y() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Z() << std::endl; */
/*   std::cout << "z vector = " << std::setw(12) << std::fixed << std::setprecision(5) << z.X() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z.Y() */
/*        << ", " << std::setw(12) << std::fixed << std::setprecision(5) <<z.Z() << std::endl; */
/* } */


/* void checkNorm ( double check ) { */
/*   double tol = 1.0e-3; */
/*   if (1.0-std::abs(check)>tol) { */
/*     std::cout << "NOT orthonormal!" << std::endl; */
/*   } else if (1.0+check<=tol) { */
/*     std::cout << "IS Left-handed (reflection)" << std::endl; */
/*   } else { */
/*     std::cout << "IS Right-handed (proper)" << std::endl; */
/*   } */

/* } */

/* int main(int argc, char *argv[]) { */
/*   std::cout << "====================== CLHEP: ========================" << std::endl; */
/*   // Examples from DD XML */
/*   // <ReflectionRotation name="180R" thetaX="90*deg" phiX="0*deg" thetaY="90*deg" phiY="90*deg" thetaZ="180*deg" phiZ="0*deg" /> */
/* {   */
/*   CLHEP::Hep3Vector x,y,z; */
/*   hepSetAxes ( x, y, z, 90*deg, 0*deg, 90*deg, 90*deg, 180*deg, 0*deg ); */
/*   CLHEP::HepRotation R; */
/*   R.rotateAxes(x, y, z);       */
/*   CLHEP::HepRotation ddr(R); */
/*   std::cout << "   *** REFLECTION *** " << std::endl; */
/*   checkNorm((x.cross(y))*z); */
/*   hepOutVecs (x, y, z); */
/*   std::cout << "Matrix output built up from vectors:" << std::endl; */
/*   hrmOut(ddr); */
/*   std::cout << "Matrix build from CLHEP::HepRep3x3 to preserve left-handedness:" << std::endl; */
/*   CLHEP::HepRep3x3 temp(x.x(),y.x(),z.x(), */
/*                  x.y(),y.y(),z.y(), */
/*                  x.z(),y.z(),z.z()); //matrix representation */
/*   CLHEP::HepRotation ddr2(temp); */
/*   hrmOut(ddr2); */
/* } */
/*   // <Rotation name="RM1509" thetaX="90*deg" phiX="-51.39999*deg" thetaY="90*deg" phiY="38.60001*deg" thetaZ="0*deg" phiZ="0*deg" /> */
/* {   */
/*   CLHEP::Hep3Vector x,y,z; */
/*   hepSetAxes ( x, y, z, 90*deg, -51.39999*deg, 90*deg, 38.60001*deg, 0*deg, 0*deg ); */
/*   CLHEP::HepRotation R; */
/*   R.rotateAxes(x, y, z); */
/*   CLHEP::HepRotation ddr(R); */
/*   std::cout << "   *** ROTATION *** " << std::endl; */
/*   checkNorm((x.cross(y))*z); */
/*   hepOutVecs (x, y, z); */
/*   std::cout << "Matrix output built up from vectors:" << std::endl; */
/*   hrmOut(ddr); */
/*   std::cout << "Matrix build from CLHEP::HepRep3x3 to preserve left-handedness:" << std::endl; */
/*   CLHEP::HepRep3x3 temp(x.x(),y.x(),z.x(), */
/*                  x.y(),y.y(),z.y(), */
/*                  x.z(),y.z(),z.z()); //matrix representation */
/*   CLHEP::HepRotation ddr2(temp); */
/*   hrmOut(ddr2); */
/* } */

/*   std::cout << "====================== ROOT::Math ========================" << std::endl; */
/*   // <ReflectionRotation name="180R" thetaX="90*deg" phiX="0*deg" thetaY="90*deg" phiY="90*deg" thetaZ="180*deg" phiZ="0*deg" /> */
/* {   */
/*   DD3Vector x,y,z; */
/*   ddSetAxes ( x, y, z, 90*deg, 0*deg, 90*deg, 90*deg, 180*deg, 0*deg ); */
/*   DDRotationMatrix R(x, y, z); */
/*   std::cout << "   *** REFLECTION *** " << std::endl; */
/*   checkNorm((x.Cross(y)).Dot(z)); */
/*   ddOutVecs (x, y, z); */
/*   std::cout << "Matrix output built up from vectors:" << std::endl; */
/*   ddRotOut(R); */
/*   std::cout << "Matrix built to preserve left-handedness:" << std::endl; */
/*   DDRotationMatrix temp(x.x(),y.x(),z.x(), */
/* 			x.y(),y.y(),z.y(), */
/* 			x.z(),y.z(),z.z()); //matrix representation */
/*   ddRotOut(temp); */
/* } */
/*   // <Rotation name="RM1509" thetaX="90*deg" phiX="-51.39999*deg" thetaY="90*deg" phiY="38.60001*deg" thetaZ="0*deg" phiZ="0*deg" /> */
/* {   */
/*   DD3Vector x,y,z; */
/*   ddSetAxes ( x, y, z, 90*deg, -51.39999*deg, 90*deg, 38.60001*deg, 0*deg, 0*deg ); */
/*   DDRotationMatrix R(x, y, z); */
/*   std::cout << "   *** ROTATION *** " << std::endl; */
/*   checkNorm( (x.Cross(y)).Dot(z) ); */
/*   ddOutVecs (x, y, z); */
/*   std::cout << "Matrix output built up from vectors:" << std::endl; */
/*   ddRotOut(R); */
/*   std::cout << "Matrix built to preserve left-handedness:" << std::endl; */
/*   DDRotationMatrix temp(x.x(),y.x(),z.x(), */
/* 			x.y(),y.y(),z.y(), */
/* 			x.z(),y.z(),z.z()); //matrix representation */
/*   ddRotOut(temp); */
/* } */


/* } */
#endif
