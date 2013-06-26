#ifndef PhysicsTools_Utilities_Abs_h
#define PhysicsTools_Utilities_Abs_h
#include <cmath>

namespace funct {

  template<typename T> 
  struct AbsStruct {
    AbsStruct(const T& t) : _(t) { }
    inline double operator()() const { return ::fabs(_()); }
    inline operator double() const { return ::fabs(_()); }
    T _; 
  };

  template<typename T> 
  struct Abs {
    typedef AbsStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template<typename T>
  inline typename Abs<T>::type abs(const T & t) { 
    return Abs<T>::compose(t); 
  }

}

#endif
