#ifndef DataFormats_GeometrySurface_SOARotation_h
#define DataFormats_GeometrySurface_SOARotation_h

template <class T> class TkRotation;

// to be moved in an external common library???

/** Rotation matrix used by SOA (as in GPU)
 */

template <class T>
class SOARotation {
public:

  constexpr inline
  SOARotation(){}

  constexpr inline
  explicit SOARotation(T) :
    R11(1), R12(0), R13(0),
    R21(0), R22(1), R23(0),
    R31(0), R32(0), R33(1) {}

  constexpr inline
  SOARotation(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    R11(xx), R12(xy), R13(xz),
    R21(yx), R22(yy), R23(yz),
    R31(zx), R32(zy), R33(zz) {}

  constexpr inline
  SOARotation(const T* p) :
    R11(p[0]), R12(p[1]), R13(p[2]),
    R21(p[3]), R22(p[4]), R23(p[5]),
    R31(p[6]), R32(p[7]), R33(p[8]) {}


  template <typename U>
  constexpr inline
  SOARotation(const TkRotation<U>& a) :
    R11(a.xx()), R12(a.xy()), R13(a.xz()),
    R21(a.yx()), R22(a.yy()), R23(a.yz()),
    R31(a.zx()), R32(a.zy()), R33(a.zz()) {}

  constexpr inline
  SOARotation transposed() const {
      return SOARotation(R11, R21, R31,
			 R12, R22, R32,
			 R13, R23, R33);
  }

  // if frame this is to local
  constexpr inline
  void multiply(T const vx, T const vy, T const vz,
                T & ux, T & uy, T & uz) const {
     ux = R11*vx + R12*vy + R13*vz;
     uy = R21*vx + R22*vy + R23*vz;
     uz = R31*vx + R32*vy + R33*vz;
  }

  // if frame this is to global
  constexpr inline
  void multiplyInverse (T const vx, T const vy, T const vz,
              T & ux, T & uy, T & uz) const {
     ux = R11*vx + R21*vy + R31*vz;
     uy = R12*vx + R22*vy + R32*vz;
     uz = R13*vx + R23*vy + R33*vz;
  }


 // if frame this is to global
 constexpr inline
 void multiplyInverse (T const vx, T const vy,
                       T & ux, T & uy, T & uz) const {
     ux = R11*vx + R21*vy;
     uy = R12*vx + R22*vy;
     uz = R13*vx + R23*vy;
  }


  constexpr inline
  T const &xx() const { return R11; }
  constexpr inline
  T const &xy() const { return R12; }
  constexpr inline
  T const &xz() const { return R13; }
  constexpr inline
  T const &yx() const { return R21; }
  constexpr inline
  T const &yy() const { return R22; }
  constexpr inline
  T const &yz() const { return R23; }
  constexpr inline
  T const &zx() const { return R31; }
  constexpr inline
  T const &zy() const { return R32; }
  constexpr inline
  T const &zz() const { return R33; }

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
     px(ix), py(iy), pz(iz), rot(irot){}

  constexpr inline
  SOARotation<T> const & rotation() const { return rot; }

  constexpr inline
  void toLocal(T const vx, T const vy,	T const vz,
               T & ux, T & uy, T & uz) const {
      rot.multiply(vx-px, vy-py, vz-pz, ux, uy, uz);
  }


  constexpr inline
  void toGlobal(T const vx, T const vy, T const vz,
                T & ux, T & uy, T & uz) const {
       rot.multiplyInverse(vx, vy, vz, ux, uy, uz);
       ux+=px; uy+=py; uz+=pz;
  }

  constexpr inline
  void toGlobal(T const vx, T const vy,
                T & ux, T & uy, T & uz) const {
       rot.multiplyInverse(vx, vy, ux, uy, uz);
       ux+=px; uy+=py; uz+=pz;
  }

  constexpr inline
  void toGlobal(
    T cxx,
    T cxy, 
    T cyy,
    T * gl) const {
   
    auto const & r = rot;
    gl[0] =  r.xx()*(r.xx()*cxx+r.yx()*cxy) + r.yx()*(r.xx()*cxy+r.yx()*cyy);
    gl[1] =  r.xx()*(r.xy()*cxx+r.yy()*cxy) + r.yx()*(r.xy()*cxy+r.yy()*cyy);
    gl[2] =  r.xy()*(r.xy()*cxx+r.yy()*cxy) + r.yy()*(r.xy()*cxy+r.yy()*cyy);
    gl[3] =  r.xx()*(r.xz()*cxx+r.yz()*cxy) + r.yx()*(r.xz()*cxy+r.yz()*cyy);
    gl[4] =  r.xy()*(r.xz()*cxx+r.yz()*cxy) + r.yy()*(r.xz()*cxy+r.yz()*cyy);
    gl[5] =  r.xz()*(r.xz()*cxx+r.yz()*cxy) + r.yz()*(r.xz()*cxy+r.yz()*cyy);
  }

  constexpr inline
  void toLocal(
    T const * ge,
    T & lxx,
    T & lxy,
    T & lyy ) const {

    auto const & r = rot;

    T cxx = ge[0]; T cyx = ge[1]; T cyy = ge[2];
    T czx = ge[3]; T czy = ge[4]; T czz = ge[5];

    lxx 
      = r.xx()*(r.xx()*cxx + r.xy()*cyx + r.xz()*czx)
      + r.xy()*(r.xx()*cyx + r.xy()*cyy + r.xz()*czy)
      + r.xz()*(r.xx()*czx + r.xy()*czy + r.xz()*czz);
    lxy
      = r.yx()*(r.xx()*cxx + r.xy()*cyx + r.xz()*czx)
      + r.yy()*(r.xx()*cyx + r.xy()*cyy + r.xz()*czy)
      + r.yz()*(r.xx()*czx + r.xy()*czy + r.xz()*czz);
    lyy
      = r.yx()*(r.yx()*cxx + r.yy()*cyx + r.yz()*czx)
      + r.yy()*(r.yx()*cyx + r.yy()*cyy + r.yz()*czy)
      + r.yz()*(r.yx()*czx + r.yy()*czy + r.yz()*czz);
  }  


  constexpr inline
  T x() const { return px; }
  constexpr inline
  T y()	const { return py; }
  constexpr inline
  T z()	const { return pz; }

private:

  T px, py, pz;
  SOARotation<T> rot;

};

#endif // DataFormats_GeometrySurface_SOARotation_h
