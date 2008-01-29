#include "PhysicsTools/Utilities/src/ExpressionBinaryOperatorSetter.h"
using namespace reco::parser;

#ifdef BOOST_SPIRIT_DEBUG 
const char op2_out<std::minus<double> >::value = '-';
const char op2_out<std::plus<double> >::value = '+';
const char op2_out<std::multiplies<double> >::value = '*';
const char op2_out<std::divides<double> >::value = '/';
const char op2_out<power_of<double> >::value = '^';
#endif
