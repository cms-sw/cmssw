#ifndef Geom_newTkRotation_H
#define Geom_newTkRotation_H

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "DataFormats/Math/interface/SSERot.h"


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
  typedef Basic3DVector<T> BasicVector;

  TkRotation( ){}
  TkRotation(  mathSSE::Rot3<T> const & irot ) : rot(irot){}
  
  TkRotation( T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    rot(xx,xy,xz, yx,yy,yz, zx, zy,zz){}

  TkRotation( const T* p) : 
    rot(p[0],p[1],p[2], 
        p[3],p[4],p[5],
        p[6],p[7],p[8]) {}
	
  TkRotation( const GlobalVector & aX, const GlobalVector & aY)  {
    
    GlobalVector uX = aX.unit();
    GlobalVector uY = aY.unit();
    GlobalVector uZ(uX.cross(uY));
    
    rot.axis[0]= uX.basicVector().v;
    rot.axis[1]= uY.basicVector().v;
    rot.axis[2]= uZ.basicVector().v;
    
  }

  TkRotation( const BasicVector & aX, const BasicVector & aY)  {
    
    BasicVector uX = aX.unit();
    BasicVector uY = aY.unit();
    BasicVector uZ(uX.cross(uY));
    
    rot.axis[0]= uX.v;
    rot.axis[1]= uY.v;
    rot.axis[2]= uZ.v;
    
  }

  
  /** Construct from global vectors of the x, y and z axes.
   *  The axes are assumed to be unit vectors forming
   *  a right-handed orthonormal basis. No checks are performed!
   */
  TkRotation( const GlobalVector & uX, const GlobalVector & uY, 
	      const GlobalVector & uZ) {
    rot.axis[0]= uX.basicVector().v;
    rot.axis[1]= uY.basicVector().v;
    rot.axis[2]= uZ.basicVector().v;
  }

  TkRotation( const BasicVector & uX, const BasicVector & uY, 
	      const BasicVector & uZ) {
    rot.axis[0]= uX.v;
    rot.axis[1]= uY.v;
    rot.axis[2]= uZ.v;
  }
  
  
  /** rotation around abritrary axis by the amount of phi:
   *  its constructed by  O^-1(z<->axis) rot_z(phi) O(z<->axis)
   *  the frame is rotated such that the z-asis corresponds to the rotation
   *  axis desired. THen it's rotated round the "new" z-axis, and then
   *  the initial transformation is "taken back" again.
   *  unfortuately I'm too stupid to describe such thing directly by 3 Euler
   *  angles.. hence I have to construckt it this way...by brute force
   */
  TkRotation( const Basic3DVector<T>& axis, T phi) :
    rot(  cos(phi), sin(phi), 0, 
	  -sin(phi), cos(phi), 0,
	  0,        0, 1) {
    
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
    rot (a.xx(), a.xy(), a.xz(), 
	 a.yx(), a.yy(), a.yz(),
	 a.zx(), a.zy(), a.zz()) {}
  
  TkRotation transposed() const {
    return rot.transpose();
  }
  
  Basic3DVector<T> rotate( const Basic3DVector<T>& v) const {
    return rot.rotate(v.v);
  }

  Basic3DVector<T> rotateBack( const Basic3DVector<T>& v) const {
    return rot.rotateBack(v.v);
  }


  Basic3DVector<T> operator*( const Basic3DVector<T>& v) const {
    return rot.rotate(v.v);
  }

  Basic3DVector<T> multiplyInverse( const Basic3DVector<T>& v) const {
    return rot.rotateBack(v.v);
  }
  
  template <class Scalar>
  Basic3DVector<Scalar> multiplyInverse( const Basic3DVector<Scalar>& v) const {
    return Basic3DVector<Scalar>( xx()*v.x() + yx()*v.y() + zx()*v.z(),
				  xy()*v.x() + yy()*v.y() + zy()*v.z(),
				  xz()*v.x() + yz()*v.y() + zz()*v.z());
  }
  
  Basic3DVector<T> operator*( const Basic2DVector<T>& v) const {
    return Basic3DVector<T>( xx()*v.x() + xy()*v.y(),
			     yx()*v.x() + yy()*v.y(),
			     zx()*v.x() + zy()*v.y());
  }
  Basic3DVector<T> multiplyInverse( const Basic2DVector<T>& v) const {
    return Basic3DVector<T>( xx()*v.x() + yx()*v.y(),
			     xy()*v.x() + yy()*v.y(),
			     xz()*v.x() + yz()*v.y());
  }
  
  
  
  TkRotation operator*( const TkRotation& b) const {
    return rot*b.rot;
  }
  TkRotation multiplyInverse( const TkRotation& b) const {
    return rot.transpose()*b.rot;
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
  
  
  Basic3DVector<T> x() const { return rot.axis[0];}
  Basic3DVector<T> y() const { return rot.axis[1];}
  Basic3DVector<T> z() const { return rot.axis[2];}
  
  T xx() const { return rot.axis[0].arr[0];} 
  T xy() const { return rot.axis[0].arr[1];} 
  T xz() const { return rot.axis[0].arr[2];} 
  T yx() const { return rot.axis[1].arr[0];} 
  T yy() const { return rot.axis[1].arr[1];} 
  T yz() const { return rot.axis[1].arr[2];} 
  T zx() const { return rot.axis[2].arr[0];} 
  T zy() const { return rot.axis[2].arr[1];} 
  T zz() const { return rot.axis[2].arr[2];} 
  
private:
  
  mathSSE::Rot3<T> rot;
  
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
  TkRotation2D(  mathSSE::Rot2<T> const & irot ) : rot(irot){}
  
  TkRotation2D( T xx, T xy, T yx, T yy) :
    rot(xx,xy, yx,yy){}

  TkRotation2D( const T* p) : 
    rot(p[0],p[1],
	p[2],p[3]) {}
	
  TkRotation2D( const BasicVector & aX)  {
    
    BasicVector uX = aX.unit();
    BasicVector uY(-uX.y(),uX.x());
    
    rot.axis[0]= uX.v;
    rot.axis[1]= uY.v;
    
  }

  
  TkRotation2D( const BasicVector & uX, const BasicVector & uY) {
    rot.axis[0]= uX.v;
    rot.axis[1]= uY.v;
  }
  
  BasicVector x() const { return rot.axis[0];}
  BasicVector y() const { return rot.axis[1];}


  TkRotation2D transposed() const {
    return rot.transpose();
  }
  
  BasicVector rotate( const BasicVector& v) const {
    return rot.rotate(v.v);
  }

  BasicVector rotateBack( const BasicVector& v) const {
    return rot.rotateBack(v.v);
  }



 private:
  
  mathSSE::Rot2<T> rot;
 
};


template<>
std::ostream & operator<< <float>( std::ostream& s, const TkRotation2D<float>& r);

template<>
std::ostream & operator<< <double>( std::ostream& s, const TkRotation2D<double>& r);


#endif




