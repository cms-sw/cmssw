#ifndef PhysicsTools_Utilities_interface_FunctionsIO_h
#define PhysicsTools_Utilities_interface_FunctionsIO_h
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Operations.h"
#include <ostream>

namespace funct {

#define PRINT_FUNCTION(FUN, NAME) \
template<typename T> \
std::ostream& operator<<(std::ostream& cout, const funct::FUN<T> & f) { \
  return cout << NAME << "(" << f._ << ")"; \
} \
 \
struct __useless_ignoreme


PRINT_FUNCTION(ExpStruct, "exp");
PRINT_FUNCTION(LogStruct, "log");
PRINT_FUNCTION(SinStruct, "sin");
PRINT_FUNCTION(CosStruct, "cos");

#undef PRINT_FUNCTION

#define PRINT_BINARY_OPERATOR(TMPL, OP) \
template<typename A, typename B> \
std::ostream& operator<<(std::ostream& cout, const funct::TMPL <A, B> & f) \
{ return cout << f._1 << OP << f._2; } \
 \
struct __useless_ignoreme

#define PRINT_UNARY_OPERATOR(TMPL, OP) \
template<typename A> \
std::ostream& operator<<(std::ostream& cout, const funct::TMPL <A> & f) \
{ return cout << OP << f._; } \
 \
struct __useless_ignoreme

PRINT_BINARY_OPERATOR(SumStruct, " + ");
PRINT_BINARY_OPERATOR(DifferenceStruct, " - ");
PRINT_BINARY_OPERATOR(ProductStruct, " ");
PRINT_BINARY_OPERATOR(RatioStruct, "/");
PRINT_UNARY_OPERATOR( MinusStruct, "-");

#undef PRINT_BINARY_OPERATOR
#undef PRINT_UNARY_OPERATOR
}

#endif
