#ifndef PhysicsTools_Utilities_Exp_h
#define PhysicsTools_Utilities_Exp_h

#include <cmath>

namespace funct {

  template <typename T>
  struct ExpStruct {
    ExpStruct(const T& t) : _(t) {}
    inline double operator()() const { return ::exp(_()); }
    T _;
  };

  template <typename T>
  struct Exp {
    typedef ExpStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Exp<T>::type exp(const T& t) {
    return Exp<T>::compose(t);
  }

}  // namespace funct

#endif
