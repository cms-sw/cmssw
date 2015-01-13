#ifndef HLTrigger_HLTfilters_TriggerExpressionParser_h
#define HLTrigger_HLTfilters_TriggerExpressionParser_h

// Note: this requires Boost 1.41 or higher, for Spirit 2.1 or higher
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>

#include "HLTrigger/HLTcore/interface/TriggerExpressionPathReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1AlgoReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1TechReader.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionOperators.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionPrescaler.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionConstant.h"

namespace triggerExpression {

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

using boost::phoenix::new_;
using boost::spirit::unused_type;

template <typename Iterator>
class Parser : public qi::grammar<Iterator, Evaluator*(), ascii::space_type>
{
public:
  Parser() :
    Parser::base_type(expression)
  {
    token_l1algo    %= qi::raw[qi::lexeme["L1_"     >> +(qi::char_("a-zA-Z0-9_*?"))]];
    token_l1tech    %= qi::raw[qi::lexeme["L1Tech_" >> +(qi::char_("a-zA-Z0-9_*?"))]];
    token_path      %= qi::raw[qi::lexeme[             +(qi::char_("a-zA-Z0-9_*?"))]];

    token            = ( qi::lit("TRUE")                [qi::_val = new_<Constant>(true)]
                       | qi::lit("FALSE")               [qi::_val = new_<Constant>(false)]
                       | token_l1algo                   [qi::_val = new_<L1AlgoReader>(qi::_1)]
                       | token_l1tech                   [qi::_val = new_<L1TechReader>(qi::_1)]
                       | token_path                     [qi::_val = new_<PathReader>(qi::_1)]
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
  typedef qi::rule<Iterator, std::string(), ascii::space_type> name_rule;
  typedef qi::rule<Iterator, Evaluator*(),  ascii::space_type> rule;

  name_rule token_l1algo;
  name_rule token_l1tech;
  name_rule token_path;

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
  bool result = qi::phrase_parse( begin, end, parser, ascii::space, evaluator );

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
  bool result = qi::phrase_parse( begin, end, parser, ascii::space, evaluator );

  if (not result or begin != end) {
    delete evaluator;
    return 0;
  }

  return evaluator;
}

} // namespace triggerExpression

#endif // HLTrigger_HLTfilters_TriggerExpressionParser_h
