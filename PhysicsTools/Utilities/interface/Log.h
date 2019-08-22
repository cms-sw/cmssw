#ifndef PhysicsTools_Utilities_Log_h
#define PhysicsTools_Utilities_Log_h

#include <cmath>

namespace funct {

  template <typename T>
  struct LogStruct {
    LogStruct(const T& t) : _(t) {}
    inline double operator()() const { return ::log(_()); }
    T _;
  };

  template <typename T>
  struct Log {
    typedef LogStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Log<T>::type log(const T& t) {
    return Log<T>::compose(t);
  }

}  // namespace funct

#endif
