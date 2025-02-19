#include <iostream>
#include <vector>
#include <cmath>
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace std;

/* 
  print a rotationmatrix in phiX, thetaX, phiY, ... representation
  when given in an (rotation-axis, rotation-angle)-representation
*/
int main()
{
  double phi,theta,alpha;
  
  cout << " axis:    phi[deg]="; cin >> phi; phi = phi*deg;
  cout << " axis:  theta[deg]="; cin >> theta; theta = theta*deg;
  cout << " angle: alpha[deg]="; cin >> alpha; alpha = alpha*deg;
  cout << endl;
  CLHEP::Hep3Vector axis;
  axis[0] = cos(phi)*sin(theta);
  axis[1] = sin(phi)*sin(theta);
  axis[2] = cos(theta);
  cout << " axis: (" << axis[0] << ',' 
                     << axis[1] << ','
		     << axis[2] << ')' << endl;
  CLHEP::HepRotation rot(axis,alpha);		      
  cout << endl;
  cout << "<Rotation name=\"YourNameHere\"" << endl;
  cout << "  phiX=\"" << rot.phiX()/deg << "*deg\"" << endl;
  cout << "  thetaX=\"" << rot.thetaX()/deg << "*deg\"" << endl;
  cout << "  phiY=\"" << rot.phiY()/deg << "*deg\"" << endl;
  cout << "  thetaY=\"" << rot.thetaY()/deg << "*deg\"" << endl;
  cout << "  phiZ=\"" << rot.phiZ()/deg << "*deg\"" << endl;
  cout << "  thetaZ=\"" << rot.thetaZ()/deg << "*deg\"/>" << endl;
  
  return 0;
}
