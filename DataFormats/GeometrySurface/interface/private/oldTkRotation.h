#ifndef Geom_oldTkRotation_H
#define Geom_oldTkRotation_H

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
/*
#include "DataFormats/GeometrySurface/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
*/
#include <iosfwd>

template <class T> class TkRotation;
template <class T> class TkRotation2D;

template <class T>
std::ostream & operator<<( std::ostream& s, const TkRotation<T>& r);
template <class T>
std::ostream & operator<<( std::ostream& s, const TkRotation2D<T>& r);

namespace geometryDetails {
  void TkRotationErr1();
  void TkRotationErr2();

}


/** Rotaion matrix used by Surface.
 */

template <class T>
class TkRotation {
public:
  typedef Vector3DBase< T, GlobalTag>  GlobalVector;

  TkRotation() : 
    R11( 1), R12( 0), R13( 0), 
    R21( 0), R22( 1), R23( 0),
    R31( 0), R32( 0), R33( 1) {}

  TkRotation( T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    R11(xx), R12(xy), R13(xz), 
    R21(yx), R22(yy), R23(yz),
    R31(zx), R32(zy), R33(zz) {}

  TkRotation( const T* p) :
    R11(p[0]), R12(p[1]), R13(p[2]), 
    R21(p[3]), R22(p[4]), R23(p[5]),
    R31(p[6]), R32(p[7]), R33(p[8]) {}

  TkRotation( const GlobalVector & aX, const GlobalVector & aY)  {

    GlobalVector uX = aX.unit();
    GlobalVector uY = aY.unit();
    GlobalVector uZ(uX.cross(uY));
 
    R11 = uX.x(); R12 = uX.y();  R13 = uX.z();
    R21 = uY.x(); R22 = uY.y();  R23 = uY.z();
    R31 = uZ.x(); R32 = uZ.y();  R33 = uZ.z();

  }

  /** Construct from global vectors of the x, y and z axes.
   *  The axes are assumed to be unit vectors forming
   *  a right-handed orthonormal basis. No checks are performed!
   */
  TkRotation( const GlobalVector & aX, const GlobalVector & aY, 
	      const GlobalVector & aZ) :
    R11( aX.x()), R12( aX.y()), R13( aX.z()), 
    R21( aY.x()), R22( aY.y()), R23( aY.z()),
    R31( aZ.x()), R32( aZ.y()), R33( aZ.z()) {}
    

  /** rotation around abritrary axis by the amount of phi:
   *  its constructed by  O^-1(z<->axis) rot_z(phi) O(z<->axis)
   *  the frame is rotated such that the z-asis corresponds to the rotation
   *  axis desired. THen it's rotated round the "new" z-axis, and then
   *  the initial transformation is "taken back" again.
   *  unfortuately I'm too stupid to describe such thing directly by 3 Euler
   *  angles.. hence I have to construckt it this way...by brute force
   */
  TkRotation( const Basic3DVector<T>& axis, T phi) :
    R11( cos(phi) ), R12( sin(phi)), R13( 0), 
    R21( -sin(phi)), R22( cos(phi)), R23( 0),
    R31( 0), R32( 0), R33( 1) {

    //rotation around the z-axis by  phi
    TkRotation tmpRotz ( cos(axis.phi()), sin(axis.phi()), 0.,
		        -sin(axis.phi()), cos(axis.phi()), 0.,
                         0.,              0.,              1. );
    //rotation around y-axis by theta
    TkRotation tmpRoty ( cos(axis.theta()), 0.,-sin(axis.theta()),
                         0.,              1.,              0.,
		         sin(axis.theta()), 0., cos(axis.theta()) );
    (*this)*=tmpRoty;
    (*this)*=tmpRotz;      // =  this * tmpRoty * tmpRotz 
   
    // (tmpRoty * tmpRotz)^-1 * this * tmpRoty * tmpRotz

    *this = (tmpRoty*tmpRotz).multiplyInverse(*this);

  }
  /* this is the same thing...derived from the CLHEP ... it gives the
     same results MODULO the sign of the rotation....  but I don't want
     that... had 
  TkRotation (const Basic3DVector<T>& axis, float phi) {
    T cx = axis.x();
    T cy = axis.y();
    T cz = axis.z();
    
    T ll = axis.mag();
    if (ll == 0) {
      geometryDetails::TkRotationErr1();
    }else{
      
      float cosa = cos(phi), sina = sin(phi);
      cx /= ll; cy /= ll; cz /= ll;   
      
       R11 = cosa + (1-cosa)*cx*cx;
       R12 =        (1-cosa)*cx*cy - sina*cz;
       R13 =        (1-cosa)*cx*cz + sina*cy;
      
       R21 =        (1-cosa)*cy*cx + sina*cz;
       R22 = cosa + (1-cosa)*cy*cy; 
       R23 =        (1-cosa)*cy*cz - sina*cx;
      
       R31 =        (1-cosa)*cz*cx - sina*cy;
       R32 =        (1-cosa)*cz*cy + sina*cx;
       R33 = cosa + (1-cosa)*cz*cz;
      
    }
    
  }
  */

  template <typename U>
  TkRotation( const TkRotation<U>& a) : 
    R11(a.xx()), R12(a.xy()), R13(a.xz()), 
    R21(a.yx()), R22(a.yy()), R23(a.yz()),
    R31(a.zx()), R32(a.zy()), R33(a.zz()) {}

  TkRotation transposed() const {
      return TkRotation( R11, R21, R31, 
			 R12, R22, R32,
			 R13, R23, R33);
  }

  Basic3DVector<T> operator*( const Basic3DVector<T>& v) const {
    return rotate(v);
  }

  Basic3DVector<T> rotate( const Basic3DVector<T>& v) const {
    return Basic3DVector<T>( R11*v.x() + R12*v.y() + R13*v.z(),
			     R21*v.x() + R22*v.y() + R23*v.z(),
			     R31*v.x() + R32*v.y() + R33*v.z());
  }

  Basic3DVector<T> multiplyInverse( const Basic3DVector<T>& v) const {
       return rotateBack(v);
   }

  Basic3DVector<T> rotateBack( const Basic3DVector<T>& v) const {
    return Basic3DVector<T>( R11*v.x() + R21*v.y() + R31*v.z(),
			     R12*v.x() + R22*v.y() + R32*v.z(),
			     R13*v.x() + R23*v.y() + R33*v.z());
  }


  Basic3DVector<T> operator*( const Basic2DVector<T>& v) const {
    return Basic3DVector<T>( R11*v.x() + R12*v.y(),
			     R21*v.x() + R22*v.y(),
			     R31*v.x() + R32*v.y());
  }
  Basic3DVector<T> multiplyInverse( const Basic2DVector<T>& v) const {
    return Basic3DVector<T>( R11*v.x() + R21*v.y(),
			     R12*v.x() + R22*v.y(),
			     R13*v.x() + R23*v.y());
  }

  

  TkRotation operator*( const TkRotation& b) const {
    return TkRotation(R11*b.R11 + R12*b.R21 + R13*b.R31,
		      R11*b.R12 + R12*b.R22 + R13*b.R32,
		      R11*b.R13 + R12*b.R23 + R13*b.R33,
		      R21*b.R11 + R22*b.R21 + R23*b.R31,
		      R21*b.R12 + R22*b.R22 + R23*b.R32,
		      R21*b.R13 + R22*b.R23 + R23*b.R33,
		      R31*b.R11 + R32*b.R21 + R33*b.R31,
		      R31*b.R12 + R32*b.R22 + R33*b.R32,
		      R31*b.R13 + R32*b.R23 + R33*b.R33);
  }

  TkRotation multiplyInverse( const TkRotation& b) const {
    return TkRotation(R11*b.R11 + R21*b.R21 + R31*b.R31,
		      R11*b.R12 + R21*b.R22 + R31*b.R32,
		      R11*b.R13 + R21*b.R23 + R31*b.R33,
		      R12*b.R11 + R22*b.R21 + R32*b.R31,
		      R12*b.R12 + R22*b.R22 + R32*b.R32,
		      R12*b.R13 + R22*b.R23 + R32*b.R33,
		      R13*b.R11 + R23*b.R21 + R33*b.R31,
		      R13*b.R12 + R23*b.R22 + R33*b.R32,
		      R13*b.R13 + R23*b.R23 + R33*b.R33);
  }

  TkRotation& operator*=( const TkRotation& b) {
    return *this = operator * (b);
  }

  // Note a *= b; <=> a = a * b; while a.transform(b); <=> a = b * a;
  TkRotation& transform(const TkRotation& b) {
    return *this = b.operator * (*this);
  }

  TkRotation & rotateAxes(const Basic3DVector<T>& newX,
			  const Basic3DVector<T>& newY,
			  const Basic3DVector<T>& newZ) {
    T del = 0.001;

    if (
	
	// the check for right-handedness is not needed since
	// we want to change the handedness when it's left in cmsim
	//
	//       fabs(newZ.x()-w.x()) > del ||
	//       fabs(newZ.y()-w.y()) > del ||
	//       fabs(newZ.z()-w.z()) > del ||
	fabs(newX.mag2()-1.) > del ||
	fabs(newY.mag2()-1.) > del || 
	fabs(newZ.mag2()-1.) > del ||
	fabs(newX.dot(newY)) > del ||
	fabs(newY.dot(newZ)) > del ||
	fabs(newZ.dot(newX)) > del) {
      geometryDetails::TkRotationErr2();
      return *this;
    } else {
      return transform(TkRotation(newX.x(), newY.x(), newZ.x(),
				  newX.y(), newY.y(), newZ.y(),
				  newX.z(), newY.z(), newZ.z()));
    }
  }

  Basic3DVector<T> x() const { return  Basic3DVector<T>(xx(),xy(),xz());}
  Basic3DVector<T> y() const { return  Basic3DVector<T>(yx(),yy(),yz());}
  Basic3DVector<T> z() const { return  Basic3DVector<T>(zx(),zy(),zz());}


  T const &xx() const { return R11;} 
  T const &xy() const { return R12;} 
  T const &xz() const { return R13;} 
  T const &yx() const { return R21;} 
  T const &yy() const { return R22;} 
  T const &yz() const { return R23;} 
  T const &zx() const { return R31;} 
  T const &zy() const { return R32;} 
  T const &zz() const { return R33;} 

private:

  T R11, R12, R13;
  T R21, R22, R23;
  T R31, R32, R33;
};


template<>
std::ostream & operator<< <float>( std::ostream& s, const TkRotation<float>& r);

template<>
std::ostream & operator<< <double>( std::ostream& s, const TkRotation<double>& r);


template <class T, class U>
inline Basic3DVector<U> operator*( const TkRotation<T>& r, const Basic3DVector<U>& v) {
  return Basic3DVector<U>( r.xx()*v.x() + r.xy()*v.y() + r.xz()*v.z(),
			   r.yx()*v.x() + r.yy()*v.y() + r.yz()*v.z(),
			   r.zx()*v.x() + r.zy()*v.y() + r.zz()*v.z());
}

template <class T, class U>
inline TkRotation<typename PreciseFloatType<T,U>::Type>
operator*( const TkRotation<T>& a, const TkRotation<U>& b) {
    typedef TkRotation<typename PreciseFloatType<T,U>::Type> RT;
    return RT( a.xx()*b.xx() + a.xy()*b.yx() + a.xz()*b.zx(),
	       a.xx()*b.xy() + a.xy()*b.yy() + a.xz()*b.zy(),
	       a.xx()*b.xz() + a.xy()*b.yz() + a.xz()*b.zz(),
	       a.yx()*b.xx() + a.yy()*b.yx() + a.yz()*b.zx(),
	       a.yx()*b.xy() + a.yy()*b.yy() + a.yz()*b.zy(),
	       a.yx()*b.xz() + a.yy()*b.yz() + a.yz()*b.zz(),
	       a.zx()*b.xx() + a.zy()*b.yx() + a.zz()*b.zx(),
	       a.zx()*b.xy() + a.zy()*b.yy() + a.zz()*b.zy(),
	       a.zx()*b.xz() + a.zy()*b.yz() + a.zz()*b.zz());
}


template <class T>
class TkRotation2D {
public:

  typedef Basic2DVector<T> BasicVector;



  TkRotation2D( ){}
  
  TkRotation2D( T xx, T xy, T yx, T yy) {
    axis[0] = BasicVector(xx,xy);
    axis[1] =BasicVector(yx, yy);
  }

  TkRotation2D( const T* p) { 
    axis[0] = BasicVector(p[0],p[1]);
    axis[1] = BasicVector(p[2],p[3]);
  }

  TkRotation2D( const BasicVector & aX)  {
    
    BasicVector uX = aX.unit();
    BasicVector uY(-uX.y(),uX.x());
    
    axis[0]= uX;
    axis[1]= uY;
    
  }

  
  TkRotation2D( const BasicVector & uX, const BasicVector & uY) {
    axis[0]= uX;
    axis[1]= uY;
  }
  
  BasicVector x() const { return axis[0];}
  BasicVector y() const { return axis[1];}


  TkRotation2D transposed() const {
    return TkRotation2D(axis[0][0], axis[1][0],
			axis[0][1], axis[1][1]
			);
  }
  
  BasicVector rotate( const BasicVector& v) const {
    return transposed().rotateBack(v);
  }

  BasicVector rotateBack( const BasicVector& v) const {
    return v[0]*axis[0] +  v[1]*axis[1];
  }



 private:
  
  BasicVector axis[2];
 
};


template<>
std::ostream & operator<< <float>( std::ostream& s, const TkRotation2D<float>& r);

template<>
std::ostream & operator<< <double>( std::ostream& s, const TkRotation2D<double>& r);


#endif




