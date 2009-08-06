//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include <iostream>

using namespace std;

int main() {
  Surface::RotationType rot;
  Surface::PositionType pos( 1, 2, 3);

  BoundPlane plane( pos, rot);

  LocalPoint lp(0,0,0);
  cout << plane.toGlobal(lp)  << ", delta = " << plane.toGlobal(lp)-pos << endl;

  double eps = 1.e-10;
  Point3DBase< float, LocalTag> f2(eps,eps,eps);
  cout << plane.toGlobal(f2) << ", delta = " << plane.toGlobal(f2)-pos << endl;
  
  Point3DBase< double, GlobalTag> nom( 1, 2, 3);
  Point3DBase< double, LocalTag> dlp(0,0,0);
  cout << plane.toGlobal(dlp) << ", delta = " << plane.toGlobal(dlp)-nom << endl;

  Point3DBase< double, LocalTag> d2(eps,eps,eps);
  cout << plane.toGlobal(d2) << ", delta = " << plane.toGlobal(d2)-nom << endl;
 
  Point3DBase< double, GlobalTag> dgp( 0.1, 0.2, 0.3);
  Vector3DBase< double, GlobalTag> dgv( 0.1, 0.2, 0.3);
  Vector3DBase< double, LocalTag> dlv( 0.1, 0.2, 0.3);

  cout << plane.toLocal(dgp) << endl;
  cout << plane.toLocal(dgv) << endl;
  cout << plane.toGlobal(dlv) << endl;

}
