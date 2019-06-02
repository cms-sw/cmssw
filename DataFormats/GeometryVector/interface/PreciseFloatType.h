#ifndef GeometryVector_PreciseFloatType_h
#define GeometryVector_PreciseFloatType_h

/** Defines a typeded corresponding to the more precise of the two 
 *  floating point types.
 *  This is useful for mixed precision arithmetic, e.g. to determin the 
 *  return type. Example of use:
 *  template <class T, class U> 
 *  typename PreciseFloatType<T,U>::Type operator+( const Vector<T>& a, 
 *                                                  const Vector<U>& b) {
 *    typename PreciseFloatType<T,U>::Type result(a); // assuming constructability
 *    return a+=b;                    // and addition for Vector of different type.
 *  }
 *
 *  This implementation is very simple, it only handles float, double and long double.
 *  for all other types PreciseFloatType<T,U>::Type is double if not compared
 *  to long double, otherwise it is long double.
 */

///  default definition is double

template <typename T, typename U>
struct PreciseFloatType {
  typedef double Type;
};

/// If the two types are identical that is also the precise type

template <typename T>
struct PreciseFloatType<T, T> {
  typedef T Type;
};

/// long double is more precise by default than other types

template <typename T>
struct PreciseFloatType<long double, T> {
  typedef long double Type;
};
template <typename T>
struct PreciseFloatType<T, long double> {
  typedef long double Type;
};
template <>
struct PreciseFloatType<long double, long double> {
  typedef long double Type;
};

#endif
