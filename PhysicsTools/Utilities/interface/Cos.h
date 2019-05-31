#ifndef PhysicsTools_Utilities_Cos_h
#define PhysicsTools_Utilities_Cos_h

#include <cmath>

namespace funct {

  template <typename T>
  struct CosStruct {
    CosStruct(const T& t) : _(t) {}
    inline double operator()() const { return ::cos(_()); }
    T _;
  };

  template <typename T>
  struct Cos {
    typedef CosStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Cos<T>::type cos(const T& t) {
    return Cos<T>::compose(t);
  }

}  // namespace funct

#endif
