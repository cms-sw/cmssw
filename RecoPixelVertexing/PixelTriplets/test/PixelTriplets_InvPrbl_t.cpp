#define private public
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitPredictionFromInvParabola.cc"
#undef private

#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include<iostream>

// oldcode
template <class T> class MappedPoint {
public:
  MappedPoint() : theU(0), theV(0), pRot(0) { }
  MappedPoint(const T & aU, const T & aV, const TkRotation<T> * aRot) 
    : theU(aU), theV(aV), pRot(aRot) { }
  MappedPoint(const Basic2DVector<T> & point, const TkRotation<T> * aRot)
    : pRot(aRot) {
    T invRadius2 = T(1)/point.mag2();
    Basic3DVector<T> rotated = (*pRot) * point;
    theU = rotated.x() * invRadius2;
    theV = rotated.y() * invRadius2;
  }
  T u() const {return theU; } 
  T v() const {return theV; }
  Basic2DVector<T> unmap () const {
    T radius2 = T(1)/(theU*theU+theV*theV); 
    Basic3DVector<T> tmp
      = (*pRot).multiplyInverse(Basic2DVector<T>(theU,theV));
    return Basic2DVector<T>( tmp.x()*radius2, tmp.y()*radius2);
  }
private:
  T theU, theV;
  const TkRotation<T> * pRot;
};

typedef  MappedPoint<double> PointUV;

void oldCode(const GlobalPoint & P1, const GlobalPoint & P2) {

  typedef TkRotation<double> Rotation;
  typedef Basic2DVector<double> Point2D;

  GlobalVector aX = GlobalVector( P1.x(), P1.y(), 0.).unit();
  GlobalVector aY( -aX.y(), aX.x(), 0.); 
  GlobalVector aZ( 0., 0., 1.);
  TkRotation<double > theRotation = Rotation(aX,aY,aZ); 

  PointUV p1(Point2D(P1.x(),P1.y()), &theRotation);
  PointUV p2(Point2D(P2.x(),P2.y()), &theRotation);

  std::cout << "\nold for " << P1 <<", " << P2 << std::endl;
  std::cout << theRotation << std::endl;
  std::cout << p1.u() << " " << p1.v() << std::endl;
  std::cout << p2.u() << " " << p2.v() << std::endl;
  std::cout << p1.unmap() << std::endl;
  std::cout << p2.unmap() << std::endl;

}

inline
Basic2DVector<double>  transform( Basic2DVector<double>  const & p, TkRotation2D<double> const &  theRotation) {
  return theRotation.rotate(p)/p.mag2();
}

inline
Basic2DVector<double>  transformBack( Basic2DVector<double>  const & p, TkRotation2D<double> const &  theRotation) {
  return theRotation.rotateBack(p)/p.mag2();
}


void newCode(const GlobalPoint & P1, const GlobalPoint & P2) {

  typedef TkRotation2D<double> Rotation;
  typedef Basic2DVector<double> Point2D;

  Rotation theRotation = Rotation(Basic2DVector<double>(P1.basicVector().xy()));
  Point2D p1 = transform(Basic2DVector<double>(P1.basicVector().xy()), theRotation);  // (1./P1.xy().mag(),0); 
  Point2D p2 = transform(Basic2DVector<double>(P2.basicVector().xy()), theRotation);

  std::cout << "\nnew for " << P1 <<", " << P2 << std::endl;
  std::cout << theRotation << std::endl;
  std::cout << p1.x() << " " << p1.y() << std::endl;
  std::cout << p2.x() << " " << p2.y() << std::endl;
  std::cout << transformBack(p1, theRotation) << std::endl;
  std::cout << transformBack(p2, theRotation) << std::endl;

}


int main() {

  GlobalPoint P1(3., 4., 7.);
  GlobalPoint P2(-2., 5., 7.);

  oldCode(P1,P2);
  newCode(P1,P2);

  oldCode(P2,P1);
  newCode(P2,P1);

  {
  ThirdHitPredictionFromInvParabola pred(P1,P2,0.2,0.05,0.1);
  std::cout << "ip min, max " <<  pred.theIpRangePlus.min() << " " << pred.theIpRangePlus.max()
	    << "  " <<  pred.theIpRangeMinus.min() << " " << pred.theIpRangeMinus.max()  << std::endl;
  std::cout << "A,B +pos " << pred.coeffA(0.1) << " " <<  pred.coeffB(0.1) << std::endl;
  std::cout << "A,B -pos " << pred.coeffA(-0.1) << " " <<  pred.coeffB(-0.1) << std::endl;

  auto rp = pred.rangeRPhi(5.,1);
  auto rn = pred.rangeRPhi(5.,-1);
  std::cout << "range " << rp.min() << " " << rp.max()
	    << " " << rn.min() << " " << rn.max() << std::endl;
  }

  ThirdHitPredictionFromInvParabola pred(-1.092805, 4.187564, -2.361283, 7.892722, 0.111413, 0.019043, 0.032000);
  std::cout << "ip min, max " <<  pred.theIpRangePlus.min() << " " << pred.theIpRangePlus.max()
	    << "  " <<  pred.theIpRangeMinus.min() << " " << pred.theIpRangeMinus.max()  << std::endl;
  {
  auto rp = pred.rangeRPhi(11.4356,1);
  auto rn = pred.rangeRPhi(11.4356,-1);
  std::cout << "range " << rp.min() << " " << rp.max()
	    << " " << rn.min() << " " << rn.max() << std::endl;
  }
  {
  auto rp = pred.rangeRPhi(13.2131,1);
  auto rn = pred.rangeRPhi(13.2131,-1);
  std::cout << "range " << rp.min() << " " << rp.max()
	    << " " << rn.min() << " " << rn.max() << std::endl;
  }

  return 0;
}
