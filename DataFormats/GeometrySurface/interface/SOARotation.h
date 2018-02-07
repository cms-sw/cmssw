#pragma once

template <class T> class TkRotation;

/** Rotation matrix used by SOA (as in GPU 
 */

template <class T>
class SOARotation {
public:

  SOARotation(){}

  explicit SOARotation(T) : 
    R11( 1), R12( 0), R13( 0), 
    R21( 0), R22( 1), R23( 0),
    R31( 0), R32( 0), R33( 1) {}

  SOARotation( T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    R11(xx), R12(xy), R13(xz), 
    R21(yx), R22(yy), R23(yz),
    R31(zx), R32(zy), R33(zz) {}

  SOARotation( const T* p) :
    R11(p[0]), R12(p[1]), R13(p[2]), 
    R21(p[3]), R22(p[4]), R23(p[5]),
    R31(p[6]), R32(p[7]), R33(p[8]) {}


  template <typename U>
  SOARotation( const TkRotation<U>& a) : 
    R11(a.xx()), R12(a.xy()), R13(a.xz()), 
    R21(a.yx()), R22(a.yy()), R23(a.yz()),
    R31(a.zx()), R32(a.zy()), R33(a.zz()) {}

  SOARotation transposed() const {
      return SOARotation( R11, R21, R31, 
			 R12, R22, R32,
			 R13, R23, R33);
  }

  // if frame this is to local
  void multiply(T const vx, T const vy, T const vz,
              T & ux, T & uy, T & uz
             ) const {
     ux = R11*vx + R12*vy + R13*vz;
     uy = R21*vx + R22*vy + R23*vz;
     uz = R31*vx + R32*vy + R33*vz;
  }

 // if frame this is to global
 void multiplyInverse (T const vx, T const vy, T const vz,
              T & ux, T & uy, T & uz
             ) const {
     ux = R11*vx + R21*vy + R31*vz;
     uy = R12*vx + R22*vy + R32*vz;  
     uz = R13*vx + R23*vy + R33*vz;
  }


 // if frame this is to global
 void multiplyInverse (T const vx, T const vy,
              T & ux, T & uy, T & uz
             ) const {
     ux = R11*vx + R21*vy;
     uy = R12*vx + R22*vy;
     uz = R13*vx + R23*vy;
  }


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


template <class T>
class SOAFrame {
public:
  
  constexpr inline   
  SOAFrame(){}
  constexpr inline
  SOAFrame(T ix, T iy, T iz, SOARotation<T> const & irot) :
     px(ix),py(iy),pz(iz){}

  constexpr inline
  SOARotation<T> const & rotation() const { return rot;}

  inline
  void toLocal(T const vx, T const vy,
              T & ux, T & uy, T & uz
             ) const {
       rot.multiplyInverse(vx,vy,ux,uy,uz);
       ux+=px; uy+=py; uz+=pz; 
  }

  constexpr inline 
  T x() const { return px;} 
  constexpr inline
  T y()	const { return py;}
  constexpr inline
  T z()	const { return pz;}

private:

  T px, py, pz;
  SOARotation<T> rot;

};
