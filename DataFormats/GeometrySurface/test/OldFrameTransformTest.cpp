#include <iostream>
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
//#include "CommonReco/RKPropagators/interface/FrameChanger.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

#include <cmath>

using namespace std;
template<typename T>
void  go() {

    typedef TkRotation<T>                   Rotation;
    typedef GloballyPositioned<T>           Frame;
    typedef typename Frame::PositionType             Position;
    typedef typename Frame::GlobalVector             GlobalVector;
    typedef typename Frame::GlobalPoint              GlobalPoint;
    typedef typename Frame::LocalVector              LocalVector;
    typedef typename Frame::LocalPoint               LocalPoint;

    std::cout << "size of Rot     " << sizeof(Rotation) << std::endl;
    std::cout << "size of Pos     " << sizeof(Position) << std::endl;
    std::cout << "size of Point   " << sizeof(GlobalPoint) << std::endl;
    std::cout << "size of Frame   " << sizeof(Frame) << std::endl;

    double a = 0.01;
    double ca = cos(a);
    double sa = sin(a);

    Rotation r1(ca, sa, 0,
		-sa, ca, 0,
		0,   0,  1);;
    Frame f1(Position(2,3,4), r1);
    cout << "f1.position() " << f1.position() << endl;
    cout << "f1.rotation() " << endl << f1.rotation() << endl;

    Rotation r2( GlobalVector( 0, 1 ,0), GlobalVector( 0, 0, 1));
    Frame f2(Position(5,6,7), r2);
    cout << "f2.position() " << f2.position() << endl;
    cout << "f2.rotation() " << endl << f2.rotation() << endl;

    // transform f2 to f1 so that f1 becomes the "global" frame of f3
    // Rotation r3 = r2.multiplyInverse(r1);
    // Rotation r3 = r2*r1;

    // Rotation r3 = r1*r2;
    // Rotation r3 = r1*r2.transposed();
    // Rotation r3 = r1.transposed()*r2;
    // Rotation r3 = r1.transposed()*r2.transposed();
    // Rotation r3 = r2*r1;
    Rotation r3 = r2*r1.transposed();


    GlobalPoint pos2(f2.position());
    LocalPoint lp3 = f1.toLocal(pos2);
    Frame f3( GlobalPoint(lp3.basicVector()), r3);
    cout << "f3.position() " << f3.position() << endl;
    cout << "f3.rotation() " << endl << f3.rotation() << endl;

// test
    GlobalPoint gp( 11,22,33);
    LocalPoint p_in1 = f1.toLocal( gp);
    typename Frame::ToLocal ff1(f1);
    LocalPoint p_in2 = f2.toLocal( gp);
    LocalPoint p_in3 = f3.toLocal( GlobalPoint(p_in1.basicVector()));
    cout << "p_in1 " << p_in1 << endl;
    cout << "p_in1 " << ff1.toLocal( gp) << endl;
    cout << "p_in2 " << p_in2 << endl;
    cout << "p_in3 " << p_in3 << endl;

    LocalPoint p_in1_from3( f3.toGlobal(p_in3).basicVector());
    cout << "p_in1_from3 + " << p_in1_from3 << endl;

    BoundPlane plane(f2.position(), f2.rotation());

//     FrameChanger<double> changer;
//     FrameChanger<double>::PlanePtr pp = changer.toFrame( plane, f1);

/*
    FrameChanger changer;
    FrameChanger::PlanePtr pp = changer.transformPlane( plane, f1);
    
    LocalPoint p_in2p = plane.toLocal( gp);
    LocalPoint p_in3p = pp->toLocal( GlobalPoint(p_in1.basicVector()));
    cout << "p_in2p " << p_in2p << endl;
    cout << "p_in3p " << p_in3p << endl;
*/


}

void cyl() {
  using T = float;
  typedef TkRotation<T>                   Rotation;
  typedef GloballyPositioned<T>           Frame;
  typedef typename Frame::PositionType             Position;
  typedef typename Frame::GlobalVector             GlobalVector;
  typedef typename Frame::GlobalPoint              GlobalPoint;
  typedef typename Frame::LocalVector              LocalVector;
  typedef typename Frame::LocalPoint               LocalPoint;
  
  // cylinder
  {
    std::cout << " Trivial Cylinder" << std::endl;
    Rotation ll( GlobalVector( 1, 0, 0), GlobalVector( 0, 1, 0));
    Position p0(0,0,0);
    Cylinder cyl(5.,p0, ll);
    Plane t = cyl.fastTangent(GlobalPoint(3.,4.,1.));
    std::cout << t.position() << '\n' << t.rotation() << std::endl;
    std::cout << t.rotation().x()*cyl.rotation().z().cross( (t.position()-cyl.position()).basicVector() ).unit() << std::endl;
  }
  
  {
    std::cout << " rotated, displaced Cylinder" << std::endl;
    Rotation ll( Basic3DVector<T>( 1, 1, 1), .3);
    Cylinder cyl(5.f, Position(2,-1,3), ll);
    Plane t = cyl.fastTangent(LocalPoint(3.,4.,1.));
    std::cout << t.position() << '\n' << t.rotation() << std::endl;
    std::cout << t.rotation().x()*cyl.rotation().z().cross( (t.position()-cyl.position()).basicVector() ).unit() << std::endl;
  }
  
}

int main() {
  
  go<float>();
  cyl();
  std::cout << std::endl;
  go<double>();

}
