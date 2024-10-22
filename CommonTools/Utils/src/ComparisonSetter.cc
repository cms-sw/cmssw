#include "CommonTools/Utils/interface/parser/ComparisonSetter.h"

#ifdef BOOST_SPIRIT_DEBUG
namespace reco {
  namespace parser {
    template <>
    const std::string cmp_out<std::less<double> >::value = "<";
    template <>
    const std::string cmp_out<std::greater<double> >::value = ">";
    template <>
    const std::string cmp_out<std::less_equal<double> >::value = "<=";
    template <>
    const std::string cmp_out<std::greater_equal<double> >::value = ">=";
    template <>
    const std::string cmp_out<std::equal_to<double> >::value = "=";
    template <>
    const std::string cmp_out<std::not_equal_to<double> >::value = "!=";
  }  // namespace parser
}  // namespace reco
#endif
