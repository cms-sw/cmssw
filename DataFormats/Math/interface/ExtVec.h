#ifndef DataFormat_Math_ExtVec_H
#define DataFormat_Math_ExtVec_H

#include<type_traits>

typedef float __attribute__( ( vector_size( 8 ) ) ) float32x2_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double __attribute__( ( vector_size( 16 ) ) ) float64x2_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float64x4_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float32x8_t;
typedef double __attribute__( ( vector_size( 64 ) ) ) float64x8_t;


// template<typename T, int N> using ExtVec =  T __attribute__( ( vector_size( N*sizeof(T) ) ) );

template<typename T, int N>
struct ExtVecTraits {
  typedef T __attribute__( ( vector_size( N*sizeof(T) ) ) ) type;
};

template<typename T, int N> using ExtVec =  typename ExtVecTraits<T,N>::type;

template<typename T> using Vec4 = ExtVec<T,4>;
template<typename T> using Vec2 = ExtVec<T,2>;

template<typename V>
inline
auto xy(V v)-> Vec2<typename  std::remove_reference<decltype(v[0])>::type> 
{ 
  typedef typename std::remove_reference<decltype(v[0])>::type T;
  return Vec2<T>{v[0],v[1]};
}
template<typename V> 
inline
auto zw(V v) -> Vec2<typename  std::remove_reference<decltype(v[0])>::type> 
{ 
  typedef typename std::remove_reference<decltype(v[0])>::type T;
  return Vec2<T>{v[2],v[3]};
}

template<typename Vec, typename F> 
inline
Vec apply(Vec v, F f) {
  typedef typename std::remove_reference<decltype(v[0])>::type T;
  constexpr int N = sizeof(Vec)/sizeof(T);
  Vec ret;
  for (int i=0;i!=N;++i) ret[i] = f(v[i]);
  return ret;
}


template<typename Vec> 
inline
Vec cross3(Vec x, Vec y) {
  //  typedef Vec4<T> Vec;
  // yz - zy, zx - xz, xy - yx, 0
  Vec x1200 = (Vec){ x[1], x[2], x[0], x[0] };
  Vec y2010 = (Vec){ y[2], y[0], y[1], y[0] };
  Vec x2010 = (Vec){ x[2], x[0], x[1], x[0] };
  Vec y1200 = (Vec){ y[1], y[2], y[0], y[0] };
  return x1200 * y2010 - x2010 * y1200;
}

template<typename V> 
inline
auto cross2(V x, V y) ->typename std::remove_reference<decltype(x[0])>::type {
  return x[0]*y[1]-x[1]*y[0];
}



/*
template<typename T> 
T dot_product(Vec4<T>  x, Vec4<T>  y) {
  auto res = x*y;
  T ret=0;
  for (int i=0;i!=4;++i) ret+=res[i];
  return ret;
}
*/

/*
template<typename V, int K> 
inline
V get1(V v) { return (V){v[K]}; }
*/

/*
template<typename T, int N> 
inline
T dot(ExtVec<T,N>  x, ExtVec<T,N>  y) {
  T ret=0;
  for (int i=0;i!=N;++i) ret+=x[i]*y[i];
  return ret;
}
*/

template<typename V>
inline
auto dot(V  x, V y) ->typename  std::remove_reference<decltype(x[0])>::type {
  typedef typename std::remove_reference<decltype(x[0])>::type T;
  constexpr int N = sizeof(V)/sizeof(T);
  T ret=0;
  for (int i=0;i!=N;++i) ret+=x[i]*y[i];
  return ret;
}


typedef Vec2<float> Vec2F;
typedef Vec4<float> Vec4F;
typedef Vec4<float> Vec3F;
typedef Vec2<double> Vec2D;
typedef Vec4<double> Vec3D;
typedef Vec4<double> Vec4D;

/*
template<typename T>
struct As3D {
  Vec4<T> const & v;
  As3D(Vec4<T> const &iv ) : v(iv){}
};
template<typename T>
inline As3D<T> as3D(Vec4<T> const &v ) { return v;}
*/

template<typename V>
struct As3D {
  V const & v;
  As3D(V const &iv ) : v(iv){}
};
template<typename V>
inline As3D<V> as3D(V const &v ) { return v;}


// rotations

template<typename T>
struct Rot3 {
  typedef Vec4<T> Vec;
  Vec  axis[3];
  
  constexpr Rot3() :
    axis{ (Vec){T(1),0,0,0},
      (Vec){0,T(1),0,0},
	(Vec){0,0,T(1),0}
  }{}
    
  constexpr Rot3( Vec4<T> ix,  Vec4<T> iy,  Vec4<T> iz) :
    axis{ix,iy,iz}{}

  constexpr Rot3( T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    axis{ (Vec){xx,xy,xz,0},
      (Vec){yx,yy,yz,0},
	(Vec){zx,zy,zz,0}
  }{}
  
  constexpr Rot3 transpose() const {
    return Rot3( axis[0][0], axis[1][0], axis[2][0],
		 axis[0][1], axis[1][1], axis[2][1],
		 axis[0][2], axis[1][2], axis[2][2]
		 );
  }
  
  constexpr Vec4<T> x() const { return axis[0];}
  constexpr Vec4<T> y() const { return axis[1];}
  constexpr Vec4<T> z() const { return axis[2];}

    // toLocal...
    constexpr Vec4<T> rotate(Vec4<T> v) const {
      return transpose().rotateBack(v);
    }


    // toGlobal...
    constexpr Vec4<T> rotateBack(Vec4<T> v) const {
      return v[0]*axis[0] +  v[1]*axis[1] + v[2]*axis[2];
    }

    Rot3 rotate(Rot3 const& r) const {
      Rot3 tr = transpose();
      return Rot3(tr.rotateBack(r.axis[0]),tr.rotateBack(r.axis[1]),tr.rotateBack(r.axis[2]));
    }

    constexpr Rot3 rotateBack(Rot3 const& r) const {
      return Rot3(rotateBack(r.axis[0]),rotateBack(r.axis[1]),rotateBack(r.axis[2]));
    }


  };

typedef Rot3<float> Rot3F;

typedef Rot3<double> Rot3D;

template<typename T>
inline constexpr Rot3<T>  operator *(Rot3<T> const & rh, Rot3<T> const & lh) {
  return lh.rotateBack(rh);
}


template<typename T>
struct Rot2 {
  typedef Vec2<T> Vec;
  Vec2<T>  axis[2];
  
  constexpr Rot2() :
    axis{ (Vec){T(1),0},
      (Vec){0,T(1)}
  }{}
    
  constexpr Rot2( Vec2<T> ix,  Vec2<T> iy) :
    axis{ix, iy}{}

  constexpr Rot2( T xx, T xy, T yx, T yy) :
    Rot2( (Vec){xx,xy},
	  (Vec){yx,yy}
	  ){}

  constexpr Rot2 transpose() const {
    return Rot2( axis[0][0], axis[1][0],
		 axis[0][1], axis[1][1]
		 );
  }

  constexpr Vec2<T> x() { return axis[0];}
  constexpr Vec2<T> y() { return axis[1];}
  
  // toLocal...
  constexpr Vec2<T> rotate(Vec2<T> v) const {
    return transpose().rotateBack(v);
  }
  

    // toGlobal...
  constexpr  Vec2<T> rotateBack(Vec2<T> v) const {
    return v[0]*axis[0] +  v[1]*axis[1];
  }
  
  Rot2 rotate(Rot2 const& r) const {
    Rot2 tr = transpose();
    return Rot2(tr.rotateBack(r.axis[0]),tr.rotateBack(r.axis[1]));
  }
  
  constexpr Rot2 rotateBack(Rot2 const& r) const {
    return Rot2(rotateBack(r.axis[0]),rotateBack(r.axis[1]));
  }
  
  
};

  typedef Rot2<float> Rot2F;

  typedef Rot2<double> Rot2D;



template<typename T>
inline  constexpr Rot2<T>  operator *(Rot2<T> const & rh, Rot2<T> const & lh) {
  return lh.rotateBack(rh);
}



#include <iosfwd>
std::ostream & operator<<(std::ostream & out, Vec2D const & v);
std::ostream & operator<<(std::ostream & out, Vec2F const & v);
std::ostream & operator<<(std::ostream & out, Vec4F const & v);
std::ostream & operator<<(std::ostream & out, Vec4D const & v);

std::ostream & operator<<(std::ostream & out, As3D<Vec4F> const & v);
std::ostream & operator<<(std::ostream & out, As3D<Vec4D> const & v);

std::ostream & operator<<(std::ostream & out, Rot3F const & v);
std::ostream & operator<<(std::ostream & out, Rot3D const & v);
std::ostream & operator<<(std::ostream & out, Rot2F const & v);
std::ostream & operator<<(std::ostream & out, Rot2D const & v);



#include <ostream>
std::ostream & operator<<(std::ostream & out,  ::Vec4F const & v) {
  return out << '(' << v[0] <<", " << v[1] <<", "<< v[2] <<", "<< v[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  ::Vec4D const & v) {
  return out << '(' << v[0] <<", " << v[1] <<", "<< v[2] <<", "<< v[3] <<')';
}
std::ostream & operator<<(std::ostream & out,  ::Vec2F const & v) {
  return out << '(' << v[0] <<", " << v[1] <<')';
}
std::ostream & operator<<(std::ostream & out,  ::Vec2D const & v) {
  return out << '(' << v[0] <<", " << v[1] <<')';
}

std::ostream & operator<<(std::ostream & out, ::As3D<Vec4F> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, ::As3D<Vec4D> const & v) {
  return out << '(' << v.v[0] <<", " << v.v[1] <<", "<< v.v[2] <<')';
}

std::ostream & operator<<(std::ostream & out, ::Rot3F const & r){
  return out << as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

std::ostream & operator<<(std::ostream & out, ::Rot3D const & r){
  return out <<  as3D(r.axis[0]) << '\n' <<  as3D(r.axis[1]) << '\n' <<  as3D(r.axis[2]);
}

std::ostream & operator<<(std::ostream & out, ::Rot2F const & r){
  return out << r.axis[0] << '\n' << r.axis[1];
}

std::ostream & operator<<(std::ostream & out, ::Rot2D const & r){
  return out << r.axis[0] << '\n' << r.axis[1];
}



#endif
