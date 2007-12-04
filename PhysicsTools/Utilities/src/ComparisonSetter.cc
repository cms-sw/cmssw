#include "PhysicsTools/Utilities/src/ComparisonSetter.h"
using namespace reco::parser;

#ifdef BOOST_SPIRIT_DEBUG 
const std::string cmp_out<std::less<double> >::value = "<";
const std::string cmp_out<std::greater<double> >::value = ">";
const std::string cmp_out<std::less_equal<double> >::value = "<=";
const std::string cmp_out<std::greater_equal<double> >::value = ">=";
const std::string cmp_out<std::equal_to<double> >::value = "=";
const std::string cmp_out<std::not_equal_to<double> >::value = "!=";
#endif
