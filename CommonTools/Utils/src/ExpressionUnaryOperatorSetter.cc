#include "CommonTools/Utils/interface/parser/ExpressionUnaryOperatorSetter.h"

using namespace reco::parser;

#ifdef BOOST_SPIRIT_DEBUG
template <>
const std::string op1_out<std::negate<double> >::value = "-";
#endif
