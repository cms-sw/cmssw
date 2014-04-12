#include <iostream>
#include <fstream>

#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

typedef CLHEP::Hep3Vector H3V;
typedef CLHEP::HepRotation HRM;

using namespace std;

void hrmOut ( const HRM& r ) {
  cout << "row 1 = " << std::setw(12) << std::fixed << std::setprecision(5) << r.xx();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.xy();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.xz() << endl;
  cout << "row 2 =  " << std::setw(12) << std::fixed << std::setprecision(5) << r.yx();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.yy();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.yz() << endl;
  cout << "row 3 = " << std::setw(12) << std::fixed << std::setprecision(5) << r.zx();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.zy();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << r.zz() << endl;

}

void hepSetAxes( H3V& x, H3V& y, H3V& z
		 , double thetaX, double phiX
		 , double thetaY, double phiY
		 , double thetaZ, double phiZ ) {
  x[0] = sin(thetaX) * cos(phiX);
  x[1] = sin(thetaX) * sin(phiX);
  x[2] = cos(thetaX);
  y[0] = sin(thetaY) * cos(phiY);
  y[1] = sin(thetaY) * sin(phiY);
  y[2] = cos(thetaY);
  z[0] = sin(thetaZ) * cos(phiZ);
  z[1] = sin(thetaZ) * sin(phiZ);
  z[2] = cos(thetaZ);
}

void hepOutVecs( const H3V& x, const H3V& y, const H3V& z ) {
  cout << "Vectors used in construction:" << endl;
  cout << "x vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[0]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[0]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[0] << endl;
  cout << "y vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[1]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[1]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[1] << endl;
  cout << "z vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x[2]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y[2]
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z[2] << endl;
}

void ddRotOut ( const DDRotationMatrix& r ) {
  double xx, xy, xz, yx, yy, yz, zx, zy, zz;
  r.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
  cout << "row 1 = " << std::setw(12) << std::fixed << std::setprecision(5) << xx;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << xy;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << xz << endl;
  cout << "row 2 =  " << std::setw(12) << std::fixed << std::setprecision(5) << yx;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << yy;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << yz << endl;
  cout << "row 3 = " << std::setw(12) << std::fixed << std::setprecision(5) << zx;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << zy;
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << zz << endl;
  cout << "USING INTERNAL << OPERATOR" << endl;
  cout << r << endl;
  cout << "USING VECTOR COMPONENTS" << endl;
  DD3Vector x, y, z;
  r.GetComponents(x,y,z);
  cout << "col 1 = " << std::setw(12) << std::fixed << std::setprecision(5) << x.X();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Y();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Z() << endl;
  cout << "col 2 =  " << std::setw(12) << std::fixed << std::setprecision(5) << y.X();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Y();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Z() << endl;
  cout << "col 3 = " << std::setw(12) << std::fixed << std::setprecision(5) << z.X();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z.Y();
  cout << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z.Z() << endl;
}

void ddSetAxes ( DD3Vector& x, DD3Vector& y,  DD3Vector& z
		 , double thetaX, double phiX
		 , double thetaY, double phiY
		 , double thetaZ, double phiZ ) {
  x.SetX(sin(thetaX) * cos(phiX));
  x.SetY(sin(thetaX) * sin(phiX));
  x.SetZ(cos(thetaX));
  y.SetX(sin(thetaY) * cos(phiY));
  y.SetY(sin(thetaY) * sin(phiY));
  y.SetZ(cos(thetaY));
  z.SetX(sin(thetaZ) * cos(phiZ));
  z.SetY(sin(thetaZ) * sin(phiZ));
  z.SetZ(cos(thetaZ));
}

void ddOutVecs( const DD3Vector& x, const DD3Vector& y, const DD3Vector& z ) {
  cout << "Vectors used in construction:" << endl;
  cout << "x vector = " << std::setw(12) << std::fixed << std::setprecision(5) << x.X()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Y()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << x.Z() << endl;
  cout << "y vector = " << std::setw(12) << std::fixed << std::setprecision(5) << y.X()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Y()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << y.Z() << endl;
  cout << "z vector = " << std::setw(12) << std::fixed << std::setprecision(5) << z.X()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) << z.Y()
       << ", " << std::setw(12) << std::fixed << std::setprecision(5) <<z.Z() << endl;
}

void checkNorm ( double check ) {
  double tol = 1.0e-3;
  if (1.0-std::abs(check)>tol) {
    cout << "NOT orthonormal!" << endl;
  } else if (1.0+check<=tol) {
    cout << "IS Left-handed (reflection)" << endl;
  } else {
    cout << "IS Right-handed (proper)" << endl;
  }

}

int main(int /*argc*/, char **/*argv[]*/) {
  cout << "====================== CLHEP: ========================" << endl;
  // Examples from DD XML
  // <ReflectionRotation name="180R" thetaX="90*deg" phiX="0*deg" thetaY="90*deg" phiY="90*deg" thetaZ="180*deg" phiZ="0*deg" />
{  
  H3V x,y,z;
  hepSetAxes ( x, y, z, 90*deg, 0*deg, 90*deg, 90*deg, 180*deg, 0*deg );
  CLHEP::HepRotation R;
  R.rotateAxes(x, y, z);      
  HRM ddr(R);
  cout << "   *** REFLECTION *** " << endl;
  checkNorm((x.cross(y))*z);
  hepOutVecs (x, y, z);
  cout << "Matrix output built up from vectors:" << endl;
  hrmOut(ddr);
  cout << "Matrix build from CLHEP::HepRep3x3 to preserve left-handedness:" << endl;
  CLHEP::HepRep3x3 temp(x.x(),y.x(),z.x(),
                 x.y(),y.y(),z.y(),
                 x.z(),y.z(),z.z()); //matrix representation
  HRM ddr2(temp);
  hrmOut(ddr2);
}
  // <Rotation name="RM1509" thetaX="90*deg" phiX="-51.39999*deg" thetaY="90*deg" phiY="38.60001*deg" thetaZ="0*deg" phiZ="0*deg" />
{  
  H3V x,y,z;
  hepSetAxes ( x, y, z, 90*deg, -51.39999*deg, 90*deg, 38.60001*deg, 0*deg, 0*deg );
  CLHEP::HepRotation R;
  R.rotateAxes(x, y, z);
  HRM ddr(R);
  cout << "   *** ROTATION *** " << endl;
  checkNorm((x.cross(y))*z);
  hepOutVecs (x, y, z);
  cout << "Matrix output built up from vectors:" << endl;
  hrmOut(ddr);
  cout << "Matrix build from CLHEP::HepRep3x3 to preserve left-handedness:" << endl;
  CLHEP::HepRep3x3 temp(x.x(),y.x(),z.x(),
                 x.y(),y.y(),z.y(),
                 x.z(),y.z(),z.z()); //matrix representation
  HRM ddr2(temp);
  hrmOut(ddr2);
}

  cout << "====================== ROOT::Math ========================" << endl;
  // <ReflectionRotation name="180R" thetaX="90*deg" phiX="0*deg" thetaY="90*deg" phiY="90*deg" thetaZ="180*deg" phiZ="0*deg" />
{  
  DD3Vector x,y,z;
  ddSetAxes ( x, y, z, 90*deg, 0*deg, 90*deg, 90*deg, 180*deg, 0*deg );
  DDRotationMatrix R(x, y, z);
  cout << "   *** REFLECTION *** " << endl;
  checkNorm((x.Cross(y)).Dot(z));
  ddOutVecs (x, y, z);
  cout << "Matrix output built up from vectors:" << endl;
  ddRotOut(R);
  cout << "Matrix built to preserve left-handedness:" << endl;
  DDRotationMatrix temp(x.x(),y.x(),z.x(),
			x.y(),y.y(),z.y(),
			x.z(),y.z(),z.z()); //matrix representation
  ddRotOut(temp);
}
  // <Rotation name="RM1509" thetaX="90*deg" phiX="-51.39999*deg" thetaY="90*deg" phiY="38.60001*deg" thetaZ="0*deg" phiZ="0*deg" />
{  
  DD3Vector x,y,z;
  ddSetAxes ( x, y, z, 90*deg, -51.39999*deg, 90*deg, 38.60001*deg, 0*deg, 0*deg );
  DDRotationMatrix R(x, y, z);
  cout << "   *** ROTATION *** " << endl;
  checkNorm( (x.Cross(y)).Dot(z) );
  ddOutVecs (x, y, z);
  cout << "Matrix output built up from vectors:" << endl;
  ddRotOut(R);
  cout << "Matrix built to preserve left-handedness:" << endl;
  DDRotationMatrix temp(x.x(),y.x(),z.x(),
			x.y(),y.y(),z.y(),
			x.z(),y.z(),z.z()); //matrix representation
  ddRotOut(temp);
}


}
