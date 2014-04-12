//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include <iostream>

using namespace std;

void st(){}
void en(){}

template<typename Scalar>
void go() {
  typedef GloballyPositioned<float>           Frame;
  Surface::RotationType rot;
  Surface::PositionType pos( .1, .2, .3);

  std::cout << "size of Frame " << sizeof(Frame) << std::endl;
  std::cout << "size of Surface " << sizeof(Surface) << std::endl;
  std::cout << "size of Plane " << sizeof(Plane) << std::endl;
  std::cout << "size of BoundPlane " << sizeof(BoundPlane) << std::endl;


  for (int i=0; i!=10; ++i) {

    cout << "pos " << pos <<std::endl;
    BoundPlane plane( pos, rot);
    cout << "prec " << plane.posPrec() <<std::endl;
    Vector3DBase<float, GlobalTag> d(1.,1.,1.);  
    for (int j=0; j!=10;++j) {
      std::cout << plane.localZclamped(pos+d) <<" ";
      d*=0.1f;
    }
    std::cout <<std::endl;

    LocalPoint lp(0,0,0);
    cout << plane.toGlobal(lp)  << ", delta = " << plane.toGlobal(lp)-pos << endl;
    
    Scalar eps = 1.e-10;
    Point3DBase< float, LocalTag> f2(eps,eps,eps);
    cout << plane.toGlobal(f2) << ", delta = " << plane.toGlobal(f2)-pos << endl;
    
    Point3DBase< Scalar, GlobalTag> nom( 1, 2, 3);
    Point3DBase< Scalar, LocalTag> dlp(0,0,0);
    cout << plane.toGlobal(dlp) << ", delta = " << plane.toGlobal(dlp)-nom << endl;
    
    Point3DBase< Scalar, LocalTag> d2(eps,eps,eps);
    cout << plane.toGlobal(d2) << ", delta = " << plane.toGlobal(d2)-nom << endl;
    
    Point3DBase< Scalar, GlobalTag> dgp( 0.1, 0.2, 0.3);
    Vector3DBase< Scalar, GlobalTag> dgv( 0.1, 0.2, 0.3);
    Vector3DBase< Scalar, LocalTag> dlv( 0.1, 0.2, 0.3);
    
    st();
    Point3DBase<Scalar, LocalTag>   p1 =  plane.toLocal(dgp);
    Vector3DBase<Scalar, LocalTag>  v1 = plane.toLocal(dgv);
    Vector3DBase<Scalar, GlobalTag> g1 = plane.toGlobal(dlv);
    en();
    cout << p1 << endl;
    cout << v1 << endl;
    cout << g1 << endl;
    
    Scalar a[3];
    Scalar v[3] = { dgv.x(), dgv.y(), dgv.z() };
    Scalar r[9] = { rot.xx(), rot.xy(), rot.xz(),
		    rot.yx(), rot.yy(), rot.yz(),
		    rot.zx(), rot.zy(), rot.zz()
    };
    
    st();
    for (int i=0; i<3; i++) {
      int j=3*i;
      a[i] = r[j]*v[0] + r[j+1]*v[1] + r[j+2]*v[2];
    }
    en();
    
    Vector3DBase<Scalar, LocalTag>  v2(a[0],a[1],a[2]);
    cout << v2 << endl;
    cout << endl;

    pos = Surface::PositionType(4*pos.x(),4*pos.y(),4*pos.z());
    
  }
    cout << endl;
    cout << endl;
}
  

int main() {
  go<float>();
  go<double>();
  return 0;
}
