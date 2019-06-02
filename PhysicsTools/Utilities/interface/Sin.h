#ifndef PhysicsTools_Utilities_Sin_h
#define PhysicsTools_Utilities_Sin_h

#include <cmath>

namespace funct {

  template <typename T>
  struct SinStruct {
    SinStruct(const T& t) : _(t) {}
    inline double operator()() const { return ::sin(_()); }
    T _;
  };

  template <typename T>
  struct Sin {
    typedef SinStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Sin<T>::type sin(const T& t) {
    return Sin<T>::compose(t);
  }

}  // namespace funct

#endif
