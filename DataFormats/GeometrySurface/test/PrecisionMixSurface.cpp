//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include <iostream>

using namespace std;

void st(){}
void en(){}


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

  st();
  Point3DBase<double, LocalTag>   p1 =  plane.toLocal(dgp);
  Vector3DBase<double, LocalTag>  v1 = plane.toLocal(dgv);
  Vector3DBase<double, GlobalTag> g1 = plane.toGlobal(dlv);
  en();
  cout << p1 << endl;
  cout << v1 << endl;
  cout << g1 << endl;

  double a[3];
  double v[3] = { dgv.x(), dgv.y(), dgv.z() };
  double r[9] = { rot.xx(), rot.xy(), rot.xz(),
		  rot.yx(), rot.yy(), rot.yz(),
		  rot.zx(), rot.zy(), rot.zz()
  };

  st();
  for (int i=0; i<3; i++) {
    int j=3*i;
    a[i] = r[j]*v[0] + r[j+1]*v[1] + r[j+2]*v[2];
  }
  en();

  Vector3DBase<double, LocalTag>  v2(a[0],a[1],a[2]);
  cout << v2 << endl;
   


}
