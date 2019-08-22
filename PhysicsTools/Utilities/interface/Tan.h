#ifndef PhysicsTools_Utilities_Tan_h
#define PhysicsTools_Utilities_Tan_h
#include <cmath>

namespace funct {

  template <typename T>
  struct TanStruct {
    TanStruct(const T& t) : _(t) {}
    inline double operator()() const { return ::tan(_()); }
    inline operator double() const { return ::tan(_()); }
    T _;
  };

  template <typename T>
  struct Tan {
    typedef TanStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Tan<T>::type tan(const T& t) {
    return Tan<T>::compose(t);
  }

}  // namespace funct

#endif
