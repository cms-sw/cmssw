#ifndef PhysicsTools_Utilities_Sqrt_h
#define PhysicsTools_Utilities_Sqrt_h
#include <cmath>

namespace funct {

  template<typename T> 
  struct SqrtStruct {
    SqrtStruct(const T& t) : _(t) { }
    inline double operator()() const { return ::sqrt(_()); }
    inline operator double() const { return ::sqrt(_()); }
    T _; 
  };

  template<typename T> 
  struct Sqrt {
    typedef SqrtStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template<typename T>
  inline typename Sqrt<T>::type sqrt(const T & t) { 
    return Sqrt<T>::compose(t); 
  }

}

#endif
