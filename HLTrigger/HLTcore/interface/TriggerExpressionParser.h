#ifndef HLTrigger_HLTfilters_TriggerExpressionParser_h
#define HLTrigger_HLTfilters_TriggerExpressionParser_h

#include <boost/version.hpp>
#if BOOST_VERSION < 104000
#error "Boost 1.40 or later is needed, for Spirit 2.0 or later"
#endif

#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>

#include "HLTrigger/HLTcore/interface/TriggerExpressionHLTReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1Reader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1TechReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionPrescaler.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionConstant.h"

namespace triggerExpression {

namespace qi { 
  // Boost 1.41 has Spirit 2.1, with all the parser components available in the spirit::qi namespace
  using namespace boost::spirit::qi;

#if BOOST_VERSION < 104100
  // Boost 1.40 has Spirit 2.0, so we need to import the needed components explicitly
  using namespace boost::spirit::arg_names;
  using boost::spirit::char_;
  using boost::spirit::wchar;
  using boost::spirit::lit;
  using boost::spirit::wlit;
  using boost::spirit::eol;
  using boost::spirit::eoi;
  using boost::spirit::bin;
  using boost::spirit::oct;
  using boost::spirit::hex;
  using boost::spirit::byte;
  using boost::spirit::word;
  using boost::spirit::dword;
  using boost::spirit::big_word;
  using boost::spirit::big_dword;
  using boost::spirit::little_word;
  using boost::spirit::little_dword;
  using boost::spirit::qword;
  using boost::spirit::big_qword;
  using boost::spirit::little_qword;
  using boost::spirit::pad;
  using boost::spirit::ushort;
  using boost::spirit::ulong;
  using boost::spirit::uint;
  using boost::spirit::uint_;
  using boost::spirit::short_;
  using boost::spirit::long_;
  using boost::spirit::int_;
  using boost::spirit::ulong_long;
  using boost::spirit::long_long;
  using boost::spirit::float_;
  using boost::spirit::double_;
  using boost::spirit::long_double;
  using boost::spirit::left_align;
  using boost::spirit::right_align;
  using boost::spirit::center;
  using boost::spirit::delimit;
  using boost::spirit::verbatim;
  using boost::spirit::none;
  using boost::spirit::eps;
  using boost::spirit::lexeme;
  using boost::spirit::lazy;
  using boost::spirit::omit;
  using boost::spirit::raw;
  using boost::spirit::stream;
  using boost::spirit::wstream;
  using boost::spirit::token;
#endif
}
namespace ascii { 
  using namespace boost::spirit::ascii; 
}

using boost::phoenix::new_;
using boost::spirit::unused_type;

template <typename Iterator>
class Parser : public qi::grammar<Iterator, Evaluator*(), ascii::space_type>
{
public:
  Parser() : 
    Parser::base_type(expression)
  {
#if BOOST_VERSION < 104100
    alnum            = +(qi::char_('a', 'z') | qi::char_('A', 'Z') | qi::char_('0', '9') | qi::char_('_', '_') | qi::char_('*', '*') | qi::char_('?', '?'));
    token_hlt       %= qi::raw[qi::lexeme["HLT_"    >> alnum]];
    token_alca      %= qi::raw[qi::lexeme["AlCa_"   >> alnum]];
    token_l1        %= qi::raw[qi::lexeme["L1_"     >> alnum]];
    token_l1tech    %= qi::raw[qi::lexeme["L1Tech_" >> alnum]];
#else
    token_hlt       %= qi::raw[qi::lexeme["HLT_"    >> +(qi::char_("a-zA-Z0-9_*?"))]];
    token_alca      %= qi::raw[qi::lexeme["AlCa_"   >> +(qi::char_("a-zA-Z0-9_*?"))]];
    token_l1        %= qi::raw[qi::lexeme["L1_"     >> +(qi::char_("a-zA-Z0-9_*?"))]];
    token_l1tech    %= qi::raw[qi::lexeme["L1Tech_" >> +(qi::char_("a-zA-Z0-9_*?"))]];
#endif

    token            = ( token_hlt                      [qi::_val = new_<HLTReader>(qi::_1)]
                       | token_alca                     [qi::_val = new_<HLTReader>(qi::_1)]
                       | token_l1                       [qi::_val = new_<L1Reader>(qi::_1)]
                       | token_l1tech                   [qi::_val = new_<L1TechReader>(qi::_1)]
                       | qi::lit("TRUE")                [qi::_val = new_<Constant>(true)]
                       | qi::lit("FALSE")               [qi::_val = new_<Constant>(false)]
                       );

    parenthesis     %= ('(' >> expression >> ')');

    element         %= (token | parenthesis);

    prescale         = (element >> '/' >> qi::uint_)    [qi::_val = new_<Prescaler> (qi::_1, qi::_2)];

    operand         %= (prescale | element);

    unary            = ( operand                        [qi::_val = qi::_1]
                       | (qi::lit("NOT") >> operand)    [qi::_val = new_<OperatorNot> (qi::_1)]
                       );

    expression       = unary                            [qi::_val = qi::_1] 
                       >> *(
                              (qi::lit("AND") >> unary) [qi::_val = new_<OperatorAnd> (qi::_val, qi::_1)] 
                            | (qi::lit("OR")  >> unary) [qi::_val = new_<OperatorOr>  (qi::_val, qi::_1)] 
                       );
  }

private:
  typedef qi::rule<Iterator, unused_type(), ascii::space_type> void_rule;
  typedef qi::rule<Iterator, std::string(), ascii::space_type> name_rule;
  typedef qi::rule<Iterator, Evaluator*(),  ascii::space_type> rule;

#if BOOST_VERSION < 104100
  void_rule alnum;
#endif
  name_rule token_hlt;
  name_rule token_alca;
  name_rule token_l1;
  name_rule token_l1tech;

  rule token;
  rule parenthesis;
  rule element;
  rule prescale;
  rule operand;
  rule unary;
  rule expression;
};


// generic interface for string-like objects
template <class T>
Evaluator * parse(const T & text) {
  typedef typename T::const_iterator Iterator;
  Parser<Iterator> parser;
  Evaluator * evaluator = 0;

  Iterator begin = text.begin();
  Iterator end   = text.end();

  // the interface of qi::phrase_parse has changed between Boost 1.40 (Spirit 2.0) and Boost 1.41 (Spirit 2.1)
#if BOOST_VERSION < 104100
  bool result = qi::phrase_parse( begin, end, parser, evaluator, ascii::space );
#else
  bool result = qi::phrase_parse( begin, end, parser, ascii::space, evaluator );
#endif

  if (not result or begin != end) {
    delete evaluator;
    return 0;
  }

  return evaluator;
}

// overloaded interface for null-terminated strings
inline Evaluator * parse(const char * text) {
  Parser<const char *> parser;
  Evaluator * evaluator = 0;

  const char * begin = text;
  const char * end   = text + strlen(text);

  // the interface of qi::phrase_parse has changed between Boost 1.40 (Spirit 2.0) and Boost 1.41 (Spirit 2.1)
#if BOOST_VERSION < 104100
  bool result = qi::phrase_parse( begin, end, parser, evaluator, ascii::space );
#else
  bool result = qi::phrase_parse( begin, end, parser, ascii::space, evaluator );
#endif

  if (not result or begin != end) {
    delete evaluator;
    return 0;
  }

  return evaluator;
}

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionParser_h
