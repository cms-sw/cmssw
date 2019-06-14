#ifndef PhysicsTools_Utilities_interface_FunctionsIO_h
#define PhysicsTools_Utilities_interface_FunctionsIO_h
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include <ostream>

namespace funct {

  template <int n>
  std::ostream& operator<<(std::ostream& cout, const Numerical<n>&) {
    return cout << n;
  }

  template <int n, int m>
  std::ostream& operator<<(std::ostream& cout, const funct::FractionStruct<n, m>&) {
    return cout << n << "/" << m;
  }

  template <int n, int m>
  std::ostream& operator<<(std::ostream& cout, const funct::MinusStruct<funct::FractionStruct<n, m> >&) {
    return cout << "-" << n << "/" << m;
  }

#define PRINT_FUNCTION(FUN, NAME)                                        \
  template <typename T>                                                  \
  std::ostream& operator<<(std::ostream& cout, const funct::FUN<T>& f) { \
    return cout << NAME << "(" << f._ << ")";                            \
  }                                                                      \
                                                                         \
  struct __useless_ignoreme

  PRINT_FUNCTION(SqrtStruct, "sqrt");
  PRINT_FUNCTION(ExpStruct, "exp");
  PRINT_FUNCTION(LogStruct, "log");
  PRINT_FUNCTION(SinStruct, "sin");
  PRINT_FUNCTION(CosStruct, "cos");
  PRINT_FUNCTION(TanStruct, "tan");
  PRINT_FUNCTION(SgnStruct, "sgn");
  PRINT_FUNCTION(AbsStruct, "abs");

#undef PRINT_FUNCTION

#define PRINT_BINARY_OPERATOR(TMPL, OP)                                      \
  template <typename A, typename B>                                          \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL<A, B>& f) { \
    return cout << f._1 << OP << f._2;                                       \
  }                                                                          \
                                                                             \
  struct __useless_ignoreme

#define PRINT_UNARY_OPERATOR(TMPL, OP)                                    \
  template <typename A>                                                   \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL<A>& f) { \
    return cout << OP << f._;                                             \
  }                                                                       \
                                                                          \
  struct __useless_ignoreme

  PRINT_BINARY_OPERATOR(SumStruct, " + ");
  PRINT_BINARY_OPERATOR(ProductStruct, " ");
  PRINT_BINARY_OPERATOR(RatioStruct, "/");
  PRINT_BINARY_OPERATOR(PowerStruct, "^");
  PRINT_UNARY_OPERATOR(MinusStruct, "-");

#undef PRINT_BINARY_OPERATOR
#undef PRINT_UNARY_OPERATOR

  template <typename A, typename B>
  std::ostream& operator<<(std::ostream& cout, const funct::SumStruct<A, funct::MinusStruct<B> >& f) {
    return cout << f._1 << " - " << f._2._;
  }

  template <typename A, typename B>
  std::ostream& operator<<(std::ostream& cout,
                           const funct::SumStruct<funct::MinusStruct<A>, funct::MinusStruct<B> >& f) {
    return cout << "- " << f._1._ << " - " << f._2._;
  }

  template <typename A, typename B>
  std::ostream& operator<<(std::ostream& cout, const funct::SumStruct<funct::MinusStruct<A>, B>& f) {
    return cout << "- " << f._1._ << " + " << f._2;
  }

  template <typename A, int n>
  std::ostream& operator<<(std::ostream& cout, const funct::SumStruct<A, funct::Numerical<n> >& f) {
    return cout << f._1 << (n >= 0 ? " + " : " - ") << ::abs(n);
  }

  template <typename A, int n>
  std::ostream& operator<<(std::ostream& cout, const funct::SumStruct<funct::MinusStruct<A>, funct::Numerical<n> >& f) {
    return cout << "- " << f._1._ << (n >= 0 ? " + " : " - ") << ::abs(n);
  }

#define PARENTHESES(TMPL1, TMPL2, OP)                                                                            \
  template <typename A, typename B, typename C>                                                                  \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL1<funct::TMPL2<A, B>, C>& f) {                   \
    return cout << "( " << f._1 << " )" << OP << f._2;                                                           \
  }                                                                                                              \
                                                                                                                 \
  template <typename A, typename B, typename C>                                                                  \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL1<C, funct::TMPL2<A, B> >& f) {                  \
    return cout << f._1 << OP << "( " << f._2 << " )";                                                           \
  }                                                                                                              \
                                                                                                                 \
  template <typename A, typename B, typename C, typename D>                                                      \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL1<funct::TMPL2<A, B>, funct::TMPL2<C, D> >& f) { \
    return cout << "( " << f._1 << " )" << OP << "( " << f._2 << " )";                                           \
  }                                                                                                              \
                                                                                                                 \
  struct __useless_ignoreme

#define PARENTHESES_FRACT(TMPL, OP)                                                                           \
  template <int n, int m, typename A>                                                                         \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL<funct::FractionStruct<n, m>, A>& f) {        \
    return cout << "( " << f._1 << " )" << OP << f._2;                                                        \
  }                                                                                                           \
                                                                                                              \
  template <int n, int m, typename A>                                                                         \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL<A, funct::FractionStruct<n, m> >& f) {       \
    return cout << f._1 << OP << "( " << f._2 << " )";                                                        \
  }                                                                                                           \
                                                                                                              \
  template <int n, int m, int k, int l>                                                                       \
  std::ostream& operator<<(std::ostream& cout,                                                                \
                           const funct::TMPL<funct::FractionStruct<n, m>, funct::FractionStruct<k, l> >& f) { \
    return cout << "( " << f._1 << " )" << OP << "( " << f._2 << " )";                                        \
  }                                                                                                           \
                                                                                                              \
  template <int n, int m, typename A>                                                                         \
  std::ostream& operator<<(std::ostream& cout,                                                                \
                           const funct::TMPL<funct::MinusStruct<funct::FractionStruct<n, m> >, A>& f) {       \
    return cout << "( " << f._1 << " )" << OP << f._2;                                                        \
  }                                                                                                           \
                                                                                                              \
  template <int n, int m, typename A>                                                                         \
  std::ostream& operator<<(std::ostream& cout,                                                                \
                           const funct::TMPL<A, funct::MinusStruct<funct::FractionStruct<n, m> > >& f) {      \
    return cout << f._1 << OP << "( " << f._2 << " )";                                                        \
  }                                                                                                           \
  struct __useless_ignoreme

#define PARENTHESES_1(TMPL1, TMPL2, OP)                                                      \
  template <typename A, typename B>                                                          \
  std::ostream& operator<<(std::ostream& cout, const funct::TMPL1<funct::TMPL2<A, B> >& f) { \
    return cout << OP << "( " << f._ << " )";                                                \
  }                                                                                          \
  struct __useless_ignoreme

  PARENTHESES(ProductStruct, SumStruct, " ");
  PARENTHESES(ProductStruct, RatioStruct, " ");
  PARENTHESES(RatioStruct, SumStruct, "/");
  PARENTHESES(RatioStruct, ProductStruct, "/");
  PARENTHESES(RatioStruct, RatioStruct, "/");

  PARENTHESES(PowerStruct, SumStruct, "^");
  PARENTHESES(PowerStruct, ProductStruct, "^");
  PARENTHESES(PowerStruct, RatioStruct, "^");

  //PARENTHESES_FRACT(ProductStruct, " ");
  PARENTHESES_FRACT(RatioStruct, "/");
  PARENTHESES_FRACT(PowerStruct, "^");

  PARENTHESES_1(MinusStruct, SumStruct, "-");
  //PARENTHESES_1(MinusStruct, RatioStruct, "-");

#undef PARENTHESES
#undef PARENTHESES_FRACT
#undef PARENTHESES_1

}  // namespace funct

#endif
