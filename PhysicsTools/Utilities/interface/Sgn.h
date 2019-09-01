#ifndef PhysicsTools_Utilities_Sgn_h
#define PhysicsTools_Utilities_Sgn_h

namespace funct {

  template <typename T>
  struct SgnStruct {
    SgnStruct(const T& t) : _(t) {}
    inline double operator()() const { return _() >= 0 ? 1 : -1; }
    inline operator double() const { return _() >= 0 ? 1 : -1; }
    T _;
  };

  template <typename T>
  struct Sgn {
    typedef SgnStruct<T> type;
    inline static type compose(const T& t) { return type(t); }
  };

  template <typename T>
  inline typename Sgn<T>::type sgn(const T& t) {
    return Sgn<T>::compose(t);
  }

}  // namespace funct

#endif
