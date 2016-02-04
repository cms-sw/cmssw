#ifndef PhysicsTools_Utilities_Variables_h
#define PhysicsTools_Utilities_Variables_h
#include "PhysicsTools/Utilities/interface/ParametricTrait.h"
#include <iostream>

#define DEFINE_VARIABLE_T(T, X, NAME) \
namespace funct { \
struct X { \
  typedef T type; \
  X() { } \
  X(const T& x) { set(x); } \
  inline operator T() const { return value; } \
  inline T operator()() const { return value; } \
  inline static const char* name() { static const char * name = NAME; return name; } \
  inline X operator=(const T& x) { set(x); return *this; } \
  inline static void set(const T& x) { value = x; } \
private: \
  static T value; \
}; \
 \
NON_PARAMETRIC(X); \
 \
inline std::ostream& operator<<(std::ostream& cout, const funct::X &) \
{ return cout << funct::X::name(); } \
 \
} \
 \
struct __useless_ignoreme

#define IMPLEMENT_VARIABLE_T(T, X) \
namespace funct { \
  T X::value; \
} \
 \
struct __useless_ignoreme \

#define DEFINE_VARIABLE(X, NAME) \
DEFINE_VARIABLE_T(double, X, NAME)

#define IMPLEMENT_VARIABLE(X) \
IMPLEMENT_VARIABLE_T(double, X)

#define DEFINE_INT_VARIABLE(X, NAME) \
DEFINE_VARIABLE_T(int, X, NAME)

#define IMPLEMENT_INT_VARIABLE(X) \
IMPLEMENT_VARIABLE_T(int, X)

DEFINE_VARIABLE(DefaultVariable, "_");
DEFINE_VARIABLE(X, "x");
DEFINE_VARIABLE(Y, "y");
DEFINE_VARIABLE(Z, "z");

#endif
